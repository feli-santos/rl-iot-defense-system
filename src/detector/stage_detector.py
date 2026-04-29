"""StageDetector — the production MLP head used by the RL agent.

Phase 4 (PLAN §A3).  29-D feature vector -> 5-class stage logits via a
small fully-connected network. This is the model the Phase-7 evaluation
calls on every step of every episode, so latency matters: the design
target in PLAN §G4.5 is ≤ 1 ms / sample on CPU.

Locked architecture (PLAN §8.D3):
    Linear(29 -> 64) -> ReLU -> Dropout(0.2)
        -> Linear(64 -> 32) -> ReLU -> Dropout(0.2)
        -> Linear(32 -> 5)
    AdamW(lr=1e-3, weight_decay=1e-4)
    CrossEntropyLoss(weight = balanced-class-weights)
    batch=512, max_epochs=20, early-stop on val-macro-F1, patience=3

Total params ≈ 4 357 → fits in any cache; CPU latency is dominated by
Python overhead, not float math.

Public API:
    StageDetector                   constructor, .fit(X, y, X_val, y_val),
                                    .predict(X), .predict_proba(X),
                                    .save(path), .from_checkpoint(path).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.detector.evaluation import macro_f1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StageDetectorConfig:
    """Locked architecture and training hyperparameters (PLAN §8.D3)."""

    num_features: int = 29
    num_classes: int = 5
    hidden_sizes: Tuple[int, ...] = (64, 32)
    dropout: float = 0.2

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 20
    patience: int = 3
    grad_clip_norm: float = 1.0
    use_class_weights: bool = True

    # Inference
    inference_batch_size: int = 4096


@dataclass
class StageDetectorRunInfo:
    """Per-run training telemetry (committed alongside the checkpoint)."""

    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    val_macro_f1_history: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_macro_f1: float = -1.0
    train_time_seconds: float = 0.0
    n_train: int = 0
    n_val: int = 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Plain MLP head — kept private; see :class:`StageDetector` for API."""

    def __init__(self, cfg: StageDetectorConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.num_features
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class StageDetector:
    """Stage-detector head: 29-D features -> 5-class stage probabilities.

    Example:
        >>> det = StageDetector()
        >>> det.fit(X_train, y_train, X_val, y_val, seed=0)
        >>> det.predict(X_test).shape
        (N,)
        >>> det.predict_proba(X_test).shape
        (N, 5)
    """

    def __init__(self, config: Optional[StageDetectorConfig] = None) -> None:
        self.config = config or StageDetectorConfig()
        self._device = torch.device("cpu")  # CPU-only by design (latency target).
        self._model: Optional[_MLP] = None
        self.run_info: StageDetectorRunInfo = StageDetectorRunInfo()

    # --------------------------------------------------------------- training

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        seed: int = 0,
        verbose: bool = True,
    ) -> "StageDetector":
        """Train the MLP with the locked schedule. Returns self."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = self.config
        if X_train.shape[1] != cfg.num_features:
            raise ValueError(
                f"X_train has {X_train.shape[1]} features but config expects "
                f"{cfg.num_features}"
            )

        self._model = _MLP(cfg).to(self._device)

        # Class weights (balanced) — normalised to sum to num_classes so the
        # absolute loss scale is comparable with weight=None.
        if cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=cfg.num_classes).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            inv = 1.0 / counts
            weights = inv * cfg.num_classes / inv.sum()
            class_weight = torch.tensor(weights, dtype=torch.float32, device=self._device)
        else:
            class_weight = None

        criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True, seed=seed)

        # Pre-compute val tensors once.
        Xv = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))
        yv_np = np.ascontiguousarray(y_val, dtype=np.int64)

        info = StageDetectorRunInfo(n_train=int(X_train.shape[0]), n_val=int(X_val.shape[0]))
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_without_improve = 0
        t0 = time.perf_counter()

        for epoch in range(1, cfg.max_epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader, optimizer, criterion, cfg.grad_clip_norm
            )

            val_logits = self._forward_in_batches(Xv, batch_size=cfg.inference_batch_size)
            val_loss = float(
                criterion(val_logits, torch.from_numpy(yv_np)).item()
            )
            val_pred = val_logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            val_f1 = macro_f1(yv_np, val_pred, num_classes=cfg.num_classes)

            info.train_loss_history.append(train_loss)
            info.val_loss_history.append(val_loss)
            info.val_macro_f1_history.append(val_f1)

            if val_f1 > info.best_val_macro_f1:
                info.best_val_macro_f1 = val_f1
                info.best_epoch = epoch
                best_state = {
                    k: v.detach().clone() for k, v in self._model.state_dict().items()
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if verbose:
                logger.info(
                    "epoch %2d  train_loss=%.4f  val_loss=%.4f  val_macroF1=%.4f%s",
                    epoch,
                    train_loss,
                    val_loss,
                    val_f1,
                    "  <-- best" if epoch == info.best_epoch else "",
                )

            if epochs_without_improve >= cfg.patience:
                if verbose:
                    logger.info(
                        "Early stop: no improvement for %d epochs", cfg.patience
                    )
                break

        info.train_time_seconds = time.perf_counter() - t0
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        self.run_info = info
        return self

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        grad_clip: float,
    ) -> float:
        assert self._model is not None
        self._model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            xb = xb.to(self._device, non_blocking=True)
            yb = yb.to(self._device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = self._model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)
        return total_loss / max(total_n, 1)

    # ------------------------------------------------------------- inference

    def predict(
        self, X: np.ndarray, *, batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Return the most likely stage id per row, shape ``(N,)`` int64."""
        proba = self.predict_proba(X, batch_size=batch_size)
        return proba.argmax(axis=1).astype(np.int64)

    def predict_proba(
        self, X: np.ndarray, *, batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Return softmax probabilities, shape ``(N, num_classes)`` float32."""
        if self._model is None:
            raise RuntimeError("StageDetector.predict_proba called before fit()")
        bs = batch_size or self.config.inference_batch_size
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        logits = self._forward_in_batches(x, batch_size=bs)
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

    def _forward_in_batches(
        self, x: torch.Tensor, *, batch_size: int
    ) -> torch.Tensor:
        """Forward in batches, no grad, return logits as a CPU tensor."""
        assert self._model is not None
        self._model.eval()
        outs: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, x.size(0), batch_size):
                xb = x[start : start + batch_size].to(self._device)
                outs.append(self._model(xb).cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, self.config.num_classes))

    # --------------------------------------------------------------- helpers

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        ds = TensorDataset(Xt, yt)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=g if shuffle else None,
            num_workers=0,
            drop_last=False,
        )

    # ----------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        """Save model weights + config + run_info to a single .pt file."""
        if self._model is None:
            raise RuntimeError("StageDetector.save called before fit()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "run_info": asdict(self.run_info),
                "state_dict": self._model.state_dict(),
            },
            path,
        )
        # Sidecar JSON for the run telemetry — easier to grep than the .pt.
        sidecar = path.with_suffix(".run_info.json")
        sidecar.write_text(json.dumps(asdict(self.run_info), indent=2))

    @classmethod
    def from_checkpoint(cls, path: Path) -> "StageDetector":
        """Re-instantiate from a checkpoint produced by :meth:`save`."""
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict["hidden_sizes"] = tuple(cfg_dict.get("hidden_sizes", (64, 32)))
        cfg = StageDetectorConfig(**cfg_dict)
        det = cls(config=cfg)
        det._model = _MLP(cfg).to(det._device)
        det._model.load_state_dict(ckpt["state_dict"])
        det._model.eval()
        if "run_info" in ckpt:
            det.run_info = StageDetectorRunInfo(**ckpt["run_info"])
        return det
