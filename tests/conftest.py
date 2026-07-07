"""
Pytest configuration and shared fixtures for RL-IoT Defense System tests.

This module provides common test fixtures used across all test modules:
- Mock dataset configurations
- Sample feature vectors
- Kill chain stage mappings
- Generator configurations
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# =============================================================================
# Kill Chain Stage Constants
# =============================================================================

KILL_CHAIN_STAGES = {
    0: "BENIGN",
    1: "RECON",
    2: "ACCESS",
    3: "MANEUVER",
    4: "IMPACT",
}

NUM_STAGES = len(KILL_CHAIN_STAGES)


# =============================================================================
# CICIoT2023 Label Mapping (Kill Chain Stage Taxonomy)
# =============================================================================

CICIOT_TO_STAGE_MAPPING = {
    # Stage 0: BENIGN
    "BenignTraffic": 0,
    # Stage 1: RECON - Information gathering
    "Recon-PortScan": 1,
    "Recon-OSScan": 1,
    "Recon-HostDiscovery": 1,
    "Recon-PingSweep": 1,
    "VulnerabilityScan": 1,
    # Stage 2: ACCESS - Exploitation & Initial Access
    "SqlInjection": 2,
    "CommandInjection": 2,
    "XSS": 2,
    "Backdoor_Malware": 2,
    "BrowserHijacking": 2,
    "Uploading_Attack": 2,
    "DictionaryBruteForce": 2,
    # Stage 3: MANEUVER - Network positioning & spoofing
    "MITM-ArpSpoofing": 3,
    "DNS_Spoofing": 3,
    "Mirai-greeth_flood": 3,
    "Mirai-greip_flood": 3,
    "Mirai-udpplain": 3,
    # Stage 4: IMPACT - Service degradation/Denial
    "DDoS-ICMP_Flood": 4,
    "DDoS-UDP_Flood": 4,
    "DDoS-TCP_Flood": 4,
    "DDoS-PSHACK_Flood": 4,
    "DDoS-SYN_Flood": 4,
    "DDoS-RSTFINFlood": 4,
    "DDoS-SynonymousIP_Flood": 4,
    "DDoS-ICMP_Fragmentation": 4,
    "DDoS-UDP_Fragmentation": 4,
    "DDoS-ACK_Fragmentation": 4,
    "DDoS-HTTP_Flood": 4,
    "DDoS-SlowLoris": 4,
    "DoS-UDP_Flood": 4,
    "DoS-TCP_Flood": 4,
    "DoS-SYN_Flood": 4,
    "DoS-HTTP_Flood": 4,
}


# =============================================================================
# Fixtures: Configuration
# =============================================================================


@pytest.fixture
def num_features() -> int:
    """Number of features in CICIoT2023 dataset (after processing)."""
    return 46


@pytest.fixture
def num_stages() -> int:
    """Number of Kill Chain stages."""
    return NUM_STAGES


# =============================================================================
# Fixtures: Temporary Paths
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for test data."""
    data_dir = tmp_path / "data" / "processed" / "ciciot2023"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# =============================================================================
# Fixtures: Mock Dataset
# =============================================================================


@pytest.fixture
def mock_dataset(temp_data_dir: Path, num_features: int) -> dict[str, Any]:
    """Create a mock processed dataset for testing.

    Creates:
    - features.npy: Random feature vectors
    - labels.npy: CICIoT2023 labels (strings)
    - scaler.joblib: Mock scaler (identity)
    - metadata.json: Dataset metadata
    - state_indices.json: Stage to row indices mapping
    """
    import json

    import joblib
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    # Create samples for each stage
    samples_per_stage = 100
    all_features = []
    all_labels = []

    for stage_id in range(NUM_STAGES):
        stage_labels = [label for label, s in CICIOT_TO_STAGE_MAPPING.items() if s == stage_id]
        for _ in range(samples_per_stage):
            all_features.append(np.random.randn(num_features).astype(np.float32))
            all_labels.append(np.random.choice(stage_labels))

    features = np.array(all_features)
    labels = np.array(all_labels)

    # Fit and save scaler
    scaler = StandardScaler()
    scaler.fit(features)
    normalized_features = scaler.transform(features)

    # Save artifacts
    np.save(temp_data_dir / "features.npy", normalized_features)
    np.save(temp_data_dir / "labels.npy", labels)
    joblib.dump(scaler, temp_data_dir / "scaler.joblib")

    # Create state indices mapping
    state_indices: dict[str, list[int]] = {str(i): [] for i in range(NUM_STAGES)}
    for idx, label in enumerate(labels):
        stage = CICIOT_TO_STAGE_MAPPING[label]
        state_indices[str(stage)].append(idx)

    with open(temp_data_dir / "state_indices.json", "w") as f:
        json.dump(state_indices, f)

    # Save metadata
    metadata = {
        "num_samples": len(features),
        "num_features": num_features,
        "num_stages": NUM_STAGES,
        "samples_per_stage": samples_per_stage,
    }
    with open(temp_data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return {
        "path": temp_data_dir,
        "features": normalized_features,
        "labels": labels,
        "state_indices": state_indices,
        "metadata": metadata,
    }
