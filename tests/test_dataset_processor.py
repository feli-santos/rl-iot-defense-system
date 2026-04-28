"""
Tests for CICIoTProcessor adversarial environment processing.

This module tests the dataset processor's ability to generate
artifacts for the Adversarial IoT Environment.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.utils.dataset_processor import CICIoTProcessor, DataProcessingConfig


class TestDataProcessorAdversarialEnv:
    """Tests for process_for_adversarial_env method."""

    @pytest.fixture
    def raw_data_dir(self, tmp_path: Path) -> Path:
        """Create a directory with mock CICIoT2023 CSV data."""
        data_dir = tmp_path / "raw" / "CICIoT2023"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock CSV with realistic columns
        np.random.seed(42)
        n_samples = 500
        
        # Create feature columns (simplified)
        data = {
            f"feature_{i}": np.random.randn(n_samples) for i in range(10)
        }
        
        # Add label column with known CICIoT2023 labels
        labels = [
            "BenignTraffic",
            "Recon-PortScan",
            "Recon-OSScan",
            "SqlInjection",
            "XSS",
            "MITM-ArpSpoofing",
            "DDoS-TCP_Flood",
            "DDoS-UDP_Flood",
            "DoS-SYN_Flood",
        ]
        data["label"] = np.random.choice(labels, n_samples)
        
        df = pd.DataFrame(data)
        df.to_csv(data_dir / "test_data.csv", index=False)
        
        return data_dir

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create output directory for processed data."""
        output = tmp_path / "processed" / "ciciot2023"
        output.mkdir(parents=True, exist_ok=True)
        return output

    @pytest.fixture
    def processor(
        self, raw_data_dir: Path, output_dir: Path
    ) -> CICIoTProcessor:
        """Create processor with test config."""
        config = DataProcessingConfig(
            dataset_path=raw_data_dir,
            output_path=output_dir,
            sample_size=500,
            sequence_length=5,
        )
        return CICIoTProcessor(config)

    def test_process_for_adversarial_env_creates_features(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """Should create features.npy file."""
        processor.process_for_adversarial_env()
        
        features_path = output_dir / "features.npy"
        assert features_path.exists()
        
        features = np.load(features_path)
        assert features.ndim == 2
        assert features.shape[0] > 0
        assert features.dtype == np.float32

    def test_process_for_adversarial_env_creates_labels(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """Should create labels.npy file."""
        processor.process_for_adversarial_env()
        
        labels_path = output_dir / "labels.npy"
        assert labels_path.exists()
        
        labels = np.load(labels_path, allow_pickle=True)
        assert len(labels) > 0

    def test_process_for_adversarial_env_creates_state_indices(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """Should create state_indices.json file."""
        processor.process_for_adversarial_env()
        
        indices_path = output_dir / "state_indices.json"
        assert indices_path.exists()
        
        with open(indices_path, "r") as f:
            state_indices = json.load(f)
        
        # Should have all 5 stages
        assert len(state_indices) == 5
        assert all(str(i) in state_indices for i in range(5))

    def test_process_for_adversarial_env_creates_scaler(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """Should create scaler.joblib file."""
        processor.process_for_adversarial_env()
        
        scaler_path = output_dir / "scaler.joblib"
        assert scaler_path.exists()

    def test_process_for_adversarial_env_creates_metadata(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """Should create metadata.json file."""
        processor.process_for_adversarial_env()
        
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        assert "num_samples" in metadata
        assert "num_features" in metadata
        assert "num_stages" in metadata
        assert metadata["num_stages"] == 5

    def test_state_indices_cover_all_samples(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """State indices should cover all samples in dataset."""
        processor.process_for_adversarial_env()
        
        features = np.load(output_dir / "features.npy")
        num_samples = len(features)
        
        with open(output_dir / "state_indices.json", "r") as f:
            state_indices = json.load(f)
        
        # Count total indices across all stages
        total_indices = sum(len(indices) for indices in state_indices.values())
        assert total_indices == num_samples

    def test_state_indices_are_valid(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """State indices should be valid row numbers."""
        processor.process_for_adversarial_env()
        
        features = np.load(output_dir / "features.npy")
        num_samples = len(features)
        
        with open(output_dir / "state_indices.json", "r") as f:
            state_indices = json.load(f)
        
        for stage_id, indices in state_indices.items():
            for idx in indices:
                assert 0 <= idx < num_samples

    def test_results_contains_stage_counts(
        self, processor: CICIoTProcessor
    ) -> None:
        """Results should include sample counts per stage."""
        results = processor.process_for_adversarial_env()
        
        assert "stage_counts" in results
        assert len(results["stage_counts"]) == 5

    def test_features_are_normalized(
        self, processor: CICIoTProcessor, output_dir: Path
    ) -> None:
        """Normalized features should have approximately zero mean."""
        processor.process_for_adversarial_env()
        
        features = np.load(output_dir / "features.npy")
        
        # Check normalization (mean close to 0, std close to 1)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        
        assert np.abs(mean).mean() < 0.1  # Mean close to 0
        assert 0.8 < std.mean() < 1.2  # Std close to 1


class TestDataProcessorWithLabelMapper:
    """Tests for label mapping integration."""

    @pytest.fixture
    def raw_data_with_all_stages(self, tmp_path: Path) -> Path:
        """Create data with samples for all Kill Chain stages."""
        data_dir = tmp_path / "raw" / "CICIoT2023"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(42)
        
        # Labels for each stage
        stage_labels = {
            0: ["BenignTraffic"] * 50,
            1: ["Recon-PortScan", "Recon-OSScan"] * 25,
            2: ["SqlInjection", "XSS", "DictionaryBruteForce"] * 17,
            3: ["MITM-ArpSpoofing", "DNS_Spoofing"] * 25,
            4: ["DDoS-TCP_Flood", "DDoS-UDP_Flood", "DoS-SYN_Flood"] * 17,
        }
        
        all_labels = []
        for labels in stage_labels.values():
            all_labels.extend(labels)
        
        n_samples = len(all_labels)
        
        data = {f"feature_{i}": np.random.randn(n_samples) for i in range(10)}
        data["label"] = all_labels
        
        df = pd.DataFrame(data)
        df.to_csv(data_dir / "test_data.csv", index=False)
        
        return data_dir

    @pytest.fixture
    def processor(
        self, raw_data_with_all_stages: Path, tmp_path: Path
    ) -> CICIoTProcessor:
        """Create processor with test config."""
        output = tmp_path / "processed"
        output.mkdir(parents=True, exist_ok=True)
        
        config = DataProcessingConfig(
            dataset_path=raw_data_with_all_stages,
            output_path=output,
            sample_size=1000,
            sequence_length=5,
        )
        return CICIoTProcessor(config)

    def test_all_stages_have_samples(
        self, processor: CICIoTProcessor
    ) -> None:
        """All 5 Kill Chain stages should have samples."""
        results = processor.process_for_adversarial_env()
        
        stage_counts = results["stage_counts"]
        
        for stage_id in range(5):
            assert stage_counts[stage_id] > 0, f"Stage {stage_id} has no samples"


class TestDataProcessorRegressionFixes:
    """Regression tests for leakage, sampling, and correlation fixes."""

    def test_smart_sampling_respects_benign_quota_and_attack_cap(self, tmp_path: Path) -> None:
        """Smart sampler should reserve benign quota and cap majority attack classes."""
        data_dir = tmp_path / "raw" / "CICIoT2023"
        out_dir = tmp_path / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        n_benign = 150
        n_attack_a = 220
        n_attack_b = 30
        total = n_benign + n_attack_a + n_attack_b

        df = pd.DataFrame({
            "feature_0": rng.normal(size=total),
            "feature_1": rng.normal(size=total),
            "label": (
                ["BenignTraffic"] * n_benign
                + ["DDoS-UDP_Flood"] * n_attack_a
                + ["Recon-PortScan"] * n_attack_b
            ),
        })
        df.to_csv(data_dir / "smart_sample.csv", index=False)

        config = DataProcessingConfig(
            dataset_path=data_dir,
            output_path=out_dir,
            sample_size=180,
            sequence_length=5,
            sampling_mode="smart_balanced",
            benign_target_count=60,
            sampling_strategy="balanced",
        )
        processor = CICIoTProcessor(config)
        raw = processor._load_raw_data()
        sampled = processor._sample_raw_data(raw)

        counts = sampled["label"].value_counts().to_dict()
        assert counts.get("BenignTraffic", 0) == 60
        # Attack budget = 120, 2 attack classes => derived cap = 60
        assert counts.get("DDoS-UDP_Flood", 0) <= 60
        assert counts.get("Recon-PortScan", 0) <= 60

    def test_correlation_filter_drops_redundant_protected_pair(self, tmp_path: Path) -> None:
        """Even when both correlated features match keep keywords, one must be dropped."""
        config = DataProcessingConfig(
            dataset_path=tmp_path,
            output_path=tmp_path,
            sample_size=100,
            sequence_length=5,
            feature_selection=True,
            correlation_threshold=0.95,
            feature_keep_keywords=["rate"],
        )
        processor = CICIoTProcessor(config)

        n = 500
        rate = np.linspace(0.0, 100.0, n)
        srate = rate.copy()  # perfect correlation with Rate
        df = pd.DataFrame(
            {
                "Rate": rate,
                "Srate": srate,
                "Other": np.random.default_rng(123).normal(size=n),
            }
        )

        filtered = processor._remove_correlated_features(
            df,
            threshold=0.95,
            keep_features={"Rate", "Srate"},
        )

        surviving = set(filtered.columns)
        assert not ({"Rate", "Srate"} <= surviving)

    def test_scaler_is_fit_on_train_split_only(self, tmp_path: Path) -> None:
        """First train block in saved features should be near standard-normal after fit."""
        data_dir = tmp_path / "raw" / "CICIoT2023"
        out_dir = tmp_path / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(7)
        n = 240
        df = pd.DataFrame(
            {
                "feature_0": rng.normal(loc=0.0, scale=1.0, size=n),
                "feature_1": rng.normal(loc=10.0, scale=5.0, size=n),
                "label": np.where(np.arange(n) % 2 == 0, "BenignTraffic", "DDoS-TCP_Flood"),
            }
        )
        df.to_csv(data_dir / "leakage_case.csv", index=False)

        config = DataProcessingConfig(
            dataset_path=data_dir,
            output_path=out_dir,
            sample_size=n,
            sequence_length=5,
            feature_selection=False,
        )
        processor = CICIoTProcessor(config)

        raw = processor._load_raw_data()
        sampled = processor._sample_raw_data(raw)
        train_raw, _, _ = processor._split_by_target(sampled, target_column="label")
        expected_train_n = len(train_raw.dropna())

        processor.process_for_adversarial_env()
        features = np.load(out_dir / "features.npy")
        train_block = features[:expected_train_n]

        train_mean = np.mean(train_block, axis=0)
        train_std = np.std(train_block, axis=0)

        assert np.abs(train_mean).mean() < 0.1
        assert 0.8 < train_std.mean() < 1.2

    def test_metadata_includes_split_and_feature_selection_info(self, tmp_path: Path) -> None:
        """Metadata should expose split counts and dropped feature diagnostics."""
        data_dir = tmp_path / "raw" / "CICIoT2023"
        out_dir = tmp_path / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(123)
        n = 300
        rate = rng.normal(loc=50.0, scale=10.0, size=n)
        srate = rate.copy()  # Perfect correlation with Rate
        label = np.where(np.arange(n) % 3 == 0, "BenignTraffic", "DDoS-TCP_Flood")

        df = pd.DataFrame(
            {
                "Rate": rate,
                "Srate": srate,
                "Feature_1": rng.normal(size=n),
                "label": label,
            }
        )
        df.to_csv(data_dir / "metadata_case.csv", index=False)

        config = DataProcessingConfig(
            dataset_path=data_dir,
            output_path=out_dir,
            sample_size=n,
            sequence_length=5,
            feature_selection=True,
            correlation_threshold=0.95,
            feature_keep_keywords=["rate"],
        )
        processor = CICIoTProcessor(config)
        processor.process_for_adversarial_env()

        with open(out_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        split_info = metadata.get("split_info", {})
        assert split_info.get("train_samples", 0) > 0
        assert split_info.get("val_samples", 0) > 0
        assert split_info.get("test_samples", 0) > 0

        fs_info = metadata.get("feature_selection_info", {})
        assert fs_info.get("enabled") is True
        dropped_corr = fs_info.get("dropped_high_correlation", [])
        assert isinstance(dropped_corr, list)
        assert len(dropped_corr) >= 1
        assert set(dropped_corr).intersection({"Rate", "Srate"})
