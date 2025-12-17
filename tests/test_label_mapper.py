"""
Tests for AbstractStateLabelMapper.

This module tests the label mapping functionality that converts
CICIoT2023 attack labels to Kill Chain abstract states (0-4).
"""

import pytest

from src.utils.label_mapper import (
    AbstractStateLabelMapper,
    KillChainStage,
    STAGE_NAMES,
)


class TestKillChainStage:
    """Tests for KillChainStage enum."""

    def test_stage_values(self) -> None:
        """Kill chain stages should have correct integer values."""
        assert KillChainStage.BENIGN.value == 0
        assert KillChainStage.RECON.value == 1
        assert KillChainStage.ACCESS.value == 2
        assert KillChainStage.MANEUVER.value == 3
        assert KillChainStage.IMPACT.value == 4

    def test_stage_count(self) -> None:
        """Should have exactly 5 kill chain stages."""
        assert len(KillChainStage) == 5

    def test_stage_names_mapping(self) -> None:
        """STAGE_NAMES should map stage IDs to readable names."""
        assert STAGE_NAMES[0] == "BENIGN"
        assert STAGE_NAMES[1] == "RECON"
        assert STAGE_NAMES[2] == "ACCESS"
        assert STAGE_NAMES[3] == "MANEUVER"
        assert STAGE_NAMES[4] == "IMPACT"


class TestAbstractStateLabelMapper:
    """Tests for AbstractStateLabelMapper class."""

    @pytest.fixture
    def mapper(self) -> AbstractStateLabelMapper:
        """Create a fresh mapper instance."""
        return AbstractStateLabelMapper()

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_initialization(self, mapper: AbstractStateLabelMapper) -> None:
        """Mapper should initialize with correct mappings."""
        assert mapper is not None
        assert len(mapper.label_to_stage) > 0

    def test_num_stages(self, mapper: AbstractStateLabelMapper) -> None:
        """Mapper should report correct number of stages."""
        assert mapper.num_stages == 5

    # =========================================================================
    # BENIGN Stage (0) Tests
    # =========================================================================

    def test_benign_mapping(self, mapper: AbstractStateLabelMapper) -> None:
        """BenignTraffic should map to stage 0."""
        assert mapper.get_stage("BenignTraffic") == KillChainStage.BENIGN
        assert mapper.get_stage_id("BenignTraffic") == 0

    # =========================================================================
    # RECON Stage (1) Tests
    # =========================================================================

    @pytest.mark.parametrize(
        "label",
        [
            "Recon-PortScan",
            "Recon-OSScan",
            "Recon-HostDiscovery",
            "Recon-PingSweep",
            "VulnerabilityScan",
        ],
    )
    def test_recon_mapping(
        self, mapper: AbstractStateLabelMapper, label: str
    ) -> None:
        """Reconnaissance attacks should map to stage 1."""
        assert mapper.get_stage(label) == KillChainStage.RECON
        assert mapper.get_stage_id(label) == 1

    # =========================================================================
    # ACCESS Stage (2) Tests
    # =========================================================================

    @pytest.mark.parametrize(
        "label",
        [
            "SqlInjection",
            "CommandInjection",
            "XSS",
            "Backdoor_Malware",
            "BrowserHijacking",
            "Uploading_Attack",
            "DictionaryBruteForce",
        ],
    )
    def test_access_mapping(
        self, mapper: AbstractStateLabelMapper, label: str
    ) -> None:
        """Exploitation/access attacks should map to stage 2."""
        assert mapper.get_stage(label) == KillChainStage.ACCESS
        assert mapper.get_stage_id(label) == 2

    # =========================================================================
    # MANEUVER Stage (3) Tests
    # =========================================================================

    @pytest.mark.parametrize(
        "label",
        [
            "MITM-ArpSpoofing",
            "DNS_Spoofing",
            "Mirai-greeth_flood",
            "Mirai-greip_flood",
        ],
    )
    def test_maneuver_mapping(
        self, mapper: AbstractStateLabelMapper, label: str
    ) -> None:
        """Network positioning/spoofing attacks should map to stage 3."""
        assert mapper.get_stage(label) == KillChainStage.MANEUVER
        assert mapper.get_stage_id(label) == 3

    # =========================================================================
    # IMPACT Stage (4) Tests
    # =========================================================================

    @pytest.mark.parametrize(
        "label",
        [
            "DDoS-ICMP_Flood",
            "DDoS-UDP_Flood",
            "DDoS-TCP_Flood",
            "DDoS-PSHACK_Flood",
            "DDoS-SYN_Flood",
            "DDoS-RSTFINFlood",
            "DDoS-SynonymousIP_Flood",
            "DDoS-ICMP_Fragmentation",
            "DDoS-UDP_Fragmentation",
            "DDoS-ACK_Fragmentation",
            "DDoS-HTTP_Flood",
            "DDoS-SlowLoris",
            "DoS-UDP_Flood",
            "DoS-TCP_Flood",
            "DoS-SYN_Flood",
            "DoS-HTTP_Flood",
        ],
    )
    def test_impact_mapping(
        self, mapper: AbstractStateLabelMapper, label: str
    ) -> None:
        """DoS/DDoS attacks should map to stage 4."""
        assert mapper.get_stage(label) == KillChainStage.IMPACT
        assert mapper.get_stage_id(label) == 4

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_unknown_label_raises_error(
        self, mapper: AbstractStateLabelMapper
    ) -> None:
        """Unknown labels should raise KeyError."""
        with pytest.raises(KeyError):
            mapper.get_stage("UnknownAttackType")

    def test_case_sensitivity(self, mapper: AbstractStateLabelMapper) -> None:
        """Label matching should be case-sensitive."""
        # Correct case works
        assert mapper.get_stage_id("BenignTraffic") == 0
        
        # Wrong case should raise
        with pytest.raises(KeyError):
            mapper.get_stage("benigntraffic")

    def test_get_stage_safe_with_default(
        self, mapper: AbstractStateLabelMapper
    ) -> None:
        """get_stage_safe should return default for unknown labels."""
        result = mapper.get_stage_safe("UnknownLabel", default=KillChainStage.BENIGN)
        assert result == KillChainStage.BENIGN

    def test_get_stage_id_safe_with_default(
        self, mapper: AbstractStateLabelMapper
    ) -> None:
        """get_stage_id_safe should return default for unknown labels."""
        result = mapper.get_stage_id_safe("UnknownLabel", default=0)
        assert result == 0

    # =========================================================================
    # Reverse Mapping Tests
    # =========================================================================

    def test_get_labels_for_stage(
        self, mapper: AbstractStateLabelMapper
    ) -> None:
        """Should return all labels mapped to a given stage."""
        benign_labels = mapper.get_labels_for_stage(KillChainStage.BENIGN)
        assert "BenignTraffic" in benign_labels
        assert len(benign_labels) == 1

        recon_labels = mapper.get_labels_for_stage(KillChainStage.RECON)
        assert "Recon-PortScan" in recon_labels
        assert len(recon_labels) == 5

        impact_labels = mapper.get_labels_for_stage(KillChainStage.IMPACT)
        assert len(impact_labels) == 16  # All DDoS and DoS variants

    def test_get_labels_for_stage_id(
        self, mapper: AbstractStateLabelMapper
    ) -> None:
        """Should return all labels for a stage by ID."""
        labels = mapper.get_labels_for_stage_id(2)  # ACCESS
        assert "SqlInjection" in labels
        assert "XSS" in labels
        assert len(labels) == 7

    # =========================================================================
    # Utility Methods Tests
    # =========================================================================

    def test_all_labels_property(self, mapper: AbstractStateLabelMapper) -> None:
        """all_labels should return list of all known labels."""
        all_labels = mapper.all_labels
        assert isinstance(all_labels, list)
        assert len(all_labels) == 33  # Total CICIoT2023 attack classes mapped
        assert "BenignTraffic" in all_labels
        assert "DDoS-ICMP_Flood" in all_labels

    def test_is_attack(self, mapper: AbstractStateLabelMapper) -> None:
        """is_attack should correctly identify attack vs benign."""
        assert mapper.is_attack("BenignTraffic") is False
        assert mapper.is_attack("DDoS-ICMP_Flood") is True
        assert mapper.is_attack("Recon-PortScan") is True

    def test_stage_name(self, mapper: AbstractStateLabelMapper) -> None:
        """get_stage_name should return human-readable stage name."""
        assert mapper.get_stage_name(0) == "BENIGN"
        assert mapper.get_stage_name(4) == "IMPACT"
        assert mapper.get_stage_name(KillChainStage.RECON) == "RECON"

    # =========================================================================
    # Batch Processing Tests
    # =========================================================================

    def test_map_labels_batch(self, mapper: AbstractStateLabelMapper) -> None:
        """Should correctly map a batch of labels to stage IDs."""
        labels = ["BenignTraffic", "Recon-PortScan", "DDoS-TCP_Flood"]
        stages = mapper.map_labels_batch(labels)
        assert stages == [0, 1, 4]

    def test_map_labels_batch_with_unknown(
        self, mapper: AbstractStateLabelMapper
    ) -> None:
        """Batch mapping with unknown labels should use default."""
        labels = ["BenignTraffic", "Unknown", "DDoS-TCP_Flood"]
        stages = mapper.map_labels_batch(labels, default=-1)
        assert stages == [0, -1, 4]

    # =========================================================================
    # Stage Distribution Tests
    # =========================================================================

    def test_stage_distribution(self, mapper: AbstractStateLabelMapper) -> None:
        """get_stage_distribution should count labels per stage."""
        distribution = mapper.get_stage_distribution()
        
        assert distribution[0] == 1   # BENIGN: 1 label
        assert distribution[1] == 5   # RECON: 5 labels
        assert distribution[2] == 7   # ACCESS: 7 labels
        assert distribution[3] == 4   # MANEUVER: 4 labels
        assert distribution[4] == 16  # IMPACT: 16 labels (DDoS + DoS)
        
        # Total should match all labels
        assert sum(distribution.values()) == 33
