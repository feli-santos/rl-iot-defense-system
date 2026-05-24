"""CNN1D — Tharewal-style 1-D convolutional baseline for detector.

The 29-D feature vector is treated as a length-29 signal with a single
channel; the network is meant to capture local correlations between
adjacent features that an MLP cannot. The architecture is *deliberately*
small (~2 K params) because a) detector is about *fair baselines* not
SOTA, and b) the comparison against StageDetector is most informative
when both models are similar size.

Locked architecture (PLAN §8.D3):
    Conv1d(1 -> 16, k=3, padding=1) -> ReLU -> MaxPool1d(2)
        -> Conv1d(16 -> 32, k=3, padding=1) -> ReLU
        -> AdaptiveAvgPool1d(1)
        -> Flatten -> Linear(32, 5)
    Same optimiser, schedule, class weighting and early-stop as the MLP
    StageDetector.

Public API mirrors StageDetector for symmetry in the entrypoint script:
    train_cnn1d(X_train, y_train, X_val, y_val, *, seed) -> CNN1D
    CNN1D(...).predict(X), .predict_proba(X), .save(path),
              .from_checkpoint(path)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
class CNN1DConfig:
    """Locked architecture and training schedule (PLAN §8.D3)."""

    num_features: int = 29
    num_classes: int = 5
    in_channels: int = 1
    out_channels_1: int = 16
    out_channels_2: int = 32
    kernel_size: int = 3

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
class CNN1DRunInfo:
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    val_macro_f1_history: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_macro_f1: float = -1.0
    train_time_seconds: float = 0.0
    n_train: int = 0
    n_val: int = 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _ConvNet(nn.Module):
    """Private 1-D conv backbone."""

    def __init__(self, cfg: CNN1DConfig) -> None:
        super().__init__()
        pad = cfg.kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.out_channels_1, cfg.kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(cfg.out_channels_1, cfg.out_channels_2, cfg.kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(cfg.out_channels_2, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (N, F) or (N, 1, F).
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CNN1D:
    """1-D conv baseline. Same I/O contract as :class:`StageDetector`."""

    def __init__(self, config: CNN1DConfig | None = None) -> None:
        self.config = config or CNN1DConfig()
        self._device = torch.device("cpu")
        self._model: _ConvNet | None = None
        self.run_info: CNN1DRunInfo = CNN1DRunInfo()

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
    ) -> CNN1D:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = self.config

        if X_train.shape[1] != cfg.num_features:
            raise ValueError(
                f"X_train has {X_train.shape[1]} features but config expects {cfg.num_features}"
            )

        self._model = _ConvNet(cfg).to(self._device)

        if cfg.use_class_weights:
            counts = np.bincount(y_train, minlength=cfg.num_classes).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            inv = 1.0 / counts
            weights = inv * cfg.num_classes / inv.sum()
            class_weight = torch.tensor(weights, dtype=torch.float32)
        else:
            class_weight = None

        criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        loader = self._make_loader(X_train, y_train, shuffle=True, seed=seed)

        Xv = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))
        yv_np = np.ascontiguousarray(y_val, dtype=np.int64)

        info = CNN1DRunInfo(n_train=int(X_train.shape[0]), n_val=int(X_val.shape[0]))
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improve = 0
        t0 = time.perf_counter()

        for epoch in range(1, cfg.max_epochs + 1):
            train_loss = self._train_one_epoch(loader, optimizer, criterion, cfg.grad_clip_norm)
            val_logits = self._forward_in_batches(Xv, batch_size=cfg.inference_batch_size)
            val_loss = float(criterion(val_logits, torch.from_numpy(yv_np)).item())
            val_pred = val_logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            val_f1 = macro_f1(yv_np, val_pred, num_classes=cfg.num_classes)

            info.train_loss_history.append(train_loss)
            info.val_loss_history.append(val_loss)
            info.val_macro_f1_history.append(val_f1)

            if val_f1 > info.best_val_macro_f1:
                info.best_val_macro_f1 = val_f1
                info.best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if verbose:
                logger.info(
                    "[CNN1D] epoch %2d  train_loss=%.4f  val_loss=%.4f  val_macroF1=%.4f%s",
                    epoch,
                    train_loss,
                    val_loss,
                    val_f1,
                    "  <-- best" if epoch == info.best_epoch else "",
                )

            if epochs_without_improve >= cfg.patience:
                if verbose:
                    logger.info(
                        "[CNN1D] Early stop after %d epochs without improvement", cfg.patience
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

    def predict(self, X: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        return self.predict_proba(X, batch_size=batch_size).argmax(axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CNN1D.predict_proba called before fit()")
        bs = batch_size or self.config.inference_batch_size
        x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        logits = self._forward_in_batches(x, batch_size=bs)
        return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

    def _forward_in_batches(self, x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        assert self._model is not None
        self._model.eval()
        outs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, x.size(0), batch_size):
                xb = x[start : start + batch_size].to(self._device)
                outs.append(self._model(xb).cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, self.config.num_classes))

    # --------------------------------------------------------------- helpers

    def _make_loader(self, X: np.ndarray, y: np.ndarray, *, shuffle: bool, seed: int) -> DataLoader:
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

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("CNN1D.save called before fit()")
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
        sidecar = path.with_suffix(".run_info.json")
        sidecar.write_text(json.dumps(asdict(self.run_info), indent=2))

    @classmethod
    def from_checkpoint(cls, path: Path) -> CNN1D:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = CNN1DConfig(**ckpt["config"])
        m = cls(config=cfg)
        m._model = _ConvNet(cfg).to(m._device)
        m._model.load_state_dict(ckpt["state_dict"])
        m._model.eval()
        if "run_info" in ckpt:
            m.run_info = CNN1DRunInfo(**ckpt["run_info"])
        return m


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


def train_cnn1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 0,
    config: CNN1DConfig | None = None,
    verbose: bool = True,
) -> CNN1D:
    """Train a :class:`CNN1D` from raw arrays. Mirror of `train_random_forest`."""
    return CNN1D(config=config).fit(X_train, y_train, X_val, y_val, seed=seed, verbose=verbose)
