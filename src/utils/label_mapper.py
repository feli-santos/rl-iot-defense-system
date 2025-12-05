"""
Abstract State Label Mapper for CICIoT2023 Dataset.

This module provides the mapping between CICIoT2023 attack labels and
the 5 Kill Chain abstract states defined in the PRD (Section 3.1).

Kill Chain Stages:
    0 - BENIGN:   Baseline system operation
    1 - RECON:    Information gathering
    2 - ACCESS:   Exploitation & Initial Access
    3 - MANEUVER: Network positioning & spoofing
    4 - IMPACT:   Service degradation/Denial
"""

from enum import IntEnum
from typing import Optional, Union


class KillChainStage(IntEnum):
    """Kill Chain stages for attack classification.
    
    These stages represent the progression of a cyberattack from
    reconnaissance to impact, following the Cyber Kill Chain model.
    """
    
    BENIGN = 0
    RECON = 1
    ACCESS = 2
    MANEUVER = 3
    IMPACT = 4


# Human-readable stage names
STAGE_NAMES: dict[int, str] = {
    0: "BENIGN",
    1: "RECON",
    2: "ACCESS",
    3: "MANEUVER",
    4: "IMPACT",
}


# CICIoT2023 label to Kill Chain stage mapping (PRD Section 3.1)
_LABEL_TO_STAGE: dict[str, KillChainStage] = {
    # Stage 0: BENIGN - Baseline system operation
    "BenignTraffic": KillChainStage.BENIGN,
    
    # Stage 1: RECON - Information gathering
    "Recon-PortScan": KillChainStage.RECON,
    "Recon-OSScan": KillChainStage.RECON,
    "Recon-HostDiscovery": KillChainStage.RECON,
    "Recon-PingSweep": KillChainStage.RECON,
    "VulnerabilityScan": KillChainStage.RECON,
    
    # Stage 2: ACCESS - Exploitation & Initial Access
    "SqlInjection": KillChainStage.ACCESS,
    "CommandInjection": KillChainStage.ACCESS,
    "XSS": KillChainStage.ACCESS,
    "Backdoor_Malware": KillChainStage.ACCESS,
    "BrowserHijacking": KillChainStage.ACCESS,
    "Uploading_Attack": KillChainStage.ACCESS,
    "DictionaryBruteForce": KillChainStage.ACCESS,
    
    # Stage 3: MANEUVER - Network positioning & spoofing
    "MITM-ArpSpoofing": KillChainStage.MANEUVER,
    "DNS_Spoofing": KillChainStage.MANEUVER,
    "Mirai-greeth_flood": KillChainStage.MANEUVER,
    "Mirai-greip_flood": KillChainStage.MANEUVER,
    
    # Stage 4: IMPACT - Service degradation/Denial (DDoS variants)
    "DDoS-ICMP_Flood": KillChainStage.IMPACT,
    "DDoS-UDP_Flood": KillChainStage.IMPACT,
    "DDoS-TCP_Flood": KillChainStage.IMPACT,
    "DDoS-PSHACK_Flood": KillChainStage.IMPACT,
    "DDoS-SYN_Flood": KillChainStage.IMPACT,
    "DDoS-RSTFINFlood": KillChainStage.IMPACT,
    "DDoS-SynonymousIP_Flood": KillChainStage.IMPACT,
    "DDoS-ICMP_Fragmentation": KillChainStage.IMPACT,
    "DDoS-UDP_Fragmentation": KillChainStage.IMPACT,
    "DDoS-ACK_Fragmentation": KillChainStage.IMPACT,
    "DDoS-HTTP_Flood": KillChainStage.IMPACT,
    "DDoS-SlowLoris": KillChainStage.IMPACT,
    
    # Stage 4: IMPACT - Service degradation/Denial (DoS variants)
    "DoS-UDP_Flood": KillChainStage.IMPACT,
    "DoS-TCP_Flood": KillChainStage.IMPACT,
    "DoS-SYN_Flood": KillChainStage.IMPACT,
    "DoS-HTTP_Flood": KillChainStage.IMPACT,
}


class AbstractStateLabelMapper:
    """Maps CICIoT2023 attack labels to Kill Chain abstract states.
    
    This class provides bidirectional mapping between the 34 CICIoT2023
    attack labels and the 5 Kill Chain stages used in the adversarial
    simulation environment.
    
    Attributes:
        label_to_stage: Dictionary mapping labels to KillChainStage.
        stage_to_labels: Dictionary mapping stages to lists of labels.
        num_stages: Number of Kill Chain stages (always 5).
    
    Example:
        >>> mapper = AbstractStateLabelMapper()
        >>> mapper.get_stage_id("DDoS-TCP_Flood")
        4
        >>> mapper.get_labels_for_stage(KillChainStage.RECON)
        ['Recon-PortScan', 'Recon-OSScan', ...]
    """
    
    def __init__(self) -> None:
        """Initialize the label mapper with predefined mappings."""
        self._label_to_stage: dict[str, KillChainStage] = _LABEL_TO_STAGE.copy()
        self._stage_to_labels: dict[KillChainStage, list[str]] = self._build_reverse_mapping()
    
    def _build_reverse_mapping(self) -> dict[KillChainStage, list[str]]:
        """Build reverse mapping from stages to labels."""
        reverse: dict[KillChainStage, list[str]] = {
            stage: [] for stage in KillChainStage
        }
        for label, stage in self._label_to_stage.items():
            reverse[stage].append(label)
        return reverse
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def label_to_stage(self) -> dict[str, KillChainStage]:
        """Get a copy of the label-to-stage mapping."""
        return self._label_to_stage.copy()
    
    @property
    def stage_to_labels(self) -> dict[KillChainStage, list[str]]:
        """Get a copy of the stage-to-labels mapping."""
        return {k: v.copy() for k, v in self._stage_to_labels.items()}
    
    @property
    def num_stages(self) -> int:
        """Get the number of Kill Chain stages."""
        return len(KillChainStage)
    
    @property
    def all_labels(self) -> list[str]:
        """Get list of all known CICIoT2023 labels."""
        return list(self._label_to_stage.keys())
    
    # =========================================================================
    # Forward Mapping (Label -> Stage)
    # =========================================================================
    
    def get_stage(self, label: str) -> KillChainStage:
        """Get the Kill Chain stage for a CICIoT2023 label.
        
        Args:
            label: CICIoT2023 attack label (case-sensitive).
        
        Returns:
            KillChainStage enum value.
        
        Raises:
            KeyError: If label is not in the mapping.
        """
        return self._label_to_stage[label]
    
    def get_stage_id(self, label: str) -> int:
        """Get the integer stage ID for a CICIoT2023 label.
        
        Args:
            label: CICIoT2023 attack label (case-sensitive).
        
        Returns:
            Integer stage ID (0-4).
        
        Raises:
            KeyError: If label is not in the mapping.
        """
        return int(self._label_to_stage[label])
    
    def get_stage_safe(
        self,
        label: str,
        default: KillChainStage = KillChainStage.BENIGN,
    ) -> KillChainStage:
        """Get stage for label with fallback default.
        
        Args:
            label: CICIoT2023 attack label.
            default: Default stage if label not found.
        
        Returns:
            KillChainStage enum value.
        """
        return self._label_to_stage.get(label, default)
    
    def get_stage_id_safe(self, label: str, default: int = 0) -> int:
        """Get integer stage ID for label with fallback default.
        
        Args:
            label: CICIoT2023 attack label.
            default: Default stage ID if label not found.
        
        Returns:
            Integer stage ID.
        """
        stage = self._label_to_stage.get(label)
        return int(stage) if stage is not None else default
    
    # =========================================================================
    # Reverse Mapping (Stage -> Labels)
    # =========================================================================
    
    def get_labels_for_stage(self, stage: KillChainStage) -> list[str]:
        """Get all labels mapped to a specific stage.
        
        Args:
            stage: KillChainStage enum value.
        
        Returns:
            List of CICIoT2023 labels.
        """
        return self._stage_to_labels[stage].copy()
    
    def get_labels_for_stage_id(self, stage_id: int) -> list[str]:
        """Get all labels mapped to a specific stage ID.
        
        Args:
            stage_id: Integer stage ID (0-4).
        
        Returns:
            List of CICIoT2023 labels.
        """
        stage = KillChainStage(stage_id)
        return self._stage_to_labels[stage].copy()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def is_attack(self, label: str) -> bool:
        """Check if a label represents an attack (non-benign).
        
        Args:
            label: CICIoT2023 attack label.
        
        Returns:
            True if the label is an attack, False if benign.
        
        Raises:
            KeyError: If label is not in the mapping.
        """
        return self._label_to_stage[label] != KillChainStage.BENIGN
    
    def get_stage_name(self, stage: Union[int, KillChainStage]) -> str:
        """Get human-readable name for a stage.
        
        Args:
            stage: Stage ID or KillChainStage enum.
        
        Returns:
            Stage name string (e.g., "IMPACT").
        """
        stage_id = int(stage)
        return STAGE_NAMES[stage_id]
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def map_labels_batch(
        self,
        labels: list[str],
        default: Optional[int] = None,
    ) -> list[int]:
        """Map a batch of labels to stage IDs.
        
        Args:
            labels: List of CICIoT2023 labels.
            default: Default value for unknown labels. If None, raises KeyError.
        
        Returns:
            List of integer stage IDs.
        
        Raises:
            KeyError: If any label is unknown and no default is provided.
        """
        if default is not None:
            return [self.get_stage_id_safe(label, default) for label in labels]
        return [self.get_stage_id(label) for label in labels]
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stage_distribution(self) -> dict[int, int]:
        """Get distribution of labels across stages.
        
        Returns:
            Dictionary mapping stage IDs to label counts.
        """
        return {
            int(stage): len(labels)
            for stage, labels in self._stage_to_labels.items()
        }
