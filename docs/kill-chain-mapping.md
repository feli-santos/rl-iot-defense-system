# Kill Chain Mapping

This system abstracts CICIoT2023 labels into 5 Kill Chain stages via `AbstractStateLabelMapper`.

## Stage definitions

| ID | Stage | Description |
|---:|:------|:------------|
| 0 | BENIGN | Baseline system operation |
| 1 | RECON | Information gathering |
| 2 | ACCESS | Exploitation & initial access |
| 3 | MANEUVER | Network positioning & spoofing |
| 4 | IMPACT | Service degradation / denial |

## Label → Stage table

### Stage 0 — BENIGN
- BenignTraffic

### Stage 1 — RECON
- Recon-PortScan
- Recon-OSScan
- Recon-HostDiscovery
- Recon-PingSweep
- VulnerabilityScan

### Stage 2 — ACCESS
- SqlInjection
- CommandInjection
- XSS
- Backdoor_Malware
- BrowserHijacking
- Uploading_Attack
- DictionaryBruteForce

### Stage 3 — MANEUVER
- MITM-ArpSpoofing
- DNS_Spoofing
- Mirai-greeth_flood
- Mirai-greip_flood
- Mirai-udpplain

### Stage 4 — IMPACT
- DDoS-ICMP_Flood
- DDoS-UDP_Flood
- DDoS-TCP_Flood
- DDoS-PSHACK_Flood
- DDoS-SYN_Flood
- DDoS-RSTFINFlood
- DDoS-SynonymousIP_Flood
- DDoS-ICMP_Fragmentation
- DDoS-UDP_Fragmentation
- DDoS-ACK_Fragmentation
- DDoS-HTTP_Flood
- DDoS-SlowLoris
- DoS-UDP_Flood
- DoS-TCP_Flood
- DoS-SYN_Flood
- DoS-HTTP_Flood

## Notes

- Mapping is **case-sensitive**.
- Unknown labels default to BENIGN only in `get_stage_id_safe` (with warnings) during data processing.
- Total mapped labels: **34**.

## Source

- `src/utils/label_mapper.py`
