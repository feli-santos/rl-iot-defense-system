# Kill Chain Mapping

This document is the **canonical defense** for the projection from the
34 fine-grained CICIoT2023 attack labels into the 5 abstract Kill Chain
stages used throughout the thesis. The projection is implemented in
`src/utils/label_mapper.py::AbstractStateLabelMapper` and is enforced as
**closed**: any new CICIoT2023 label encountered at processing time
raises an explicit `KeyError`
(`tests/test_label_mapper.py::TestStringToStageIds::test_raises_on_unknown_label`).
Deletions, renames, or additions to this file MUST be mirrored in the
mapper, the test, the closed-mapping check, and the `dataset_card.md` §3
table; the four artefacts must stay in lock-step.

## 1 — Why a 5-stage abstraction?

CICIoT2023's 34 attack labels are at a level of granularity at which (a)
no per-class statistical conclusion is robust given the per-class row
budget after rebalancing (~12 121 rows for the typical class — see
`docs/dataset_card.md` §2), and (b) a defender's recommended action is
identical for many semantically related labels (a TCP SYN flood and a
UDP flood call for the same response). The Kill Chain abstraction
collapses this into 5 stages **monotone in attacker progression
severity**, which is the property the thesis's proportional-defense
reward shape (`docs/reward-shaping.md`) actually depends on.

The five stages adopted here trace to Lockheed Martin's Cyber Kill
Chain (Hutchins, Cloppert, Amin, 2011) and the closely related MITRE
ATT&CK tactic groupings, but are **deliberately coarsened** to fit the
classes that CICIoT2023 actually contains. The original Lockheed
seven-step chain (Reconnaissance / Weaponization / Delivery /
Exploitation / Installation / Command-and-Control / Actions on
Objectives) does not map cleanly because CICIoT2023 has no labelled
"Weaponization", "Installation", or "C2" classes — those phases are
either invisible at the flow-feature level or simply absent from the
testbed traces. The 5-stage mapping below is the simplest abstraction
that (i) covers every CICIoT2023 label, (ii) is monotone in attacker
progress, and (iii) admits a non-trivial recommended-action mapping
(BENIGN→OBSERVE, RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK,
IMPACT→ISOLATE — see `docs/reward-shaping.md` §Recommended actions).

## 2 — Stage definitions (operational)

Every CICIoT2023 label is assigned to **exactly one** of the five
stages below. The integer IDs are exported as
`src/utils/label_mapper.py::KillChainStage(IntEnum)` and are stable
across the thesis (used in observation vectors, reward calculations,
and confusion-matrix axes).

| ID | Short | Stage    | Operational definition (this thesis)                                                                                                                                                                                                                                                                                                                              |
|---:|:------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 |   B   | BENIGN   | Baseline non-attack traffic. The defender's reference distribution.                                                                                                                                                                                                                                                                                               |
|  1 |   R   | RECON    | **Pre-access information gathering** that does not yet attempt to exploit a vulnerability or hijack a session. Probes the existence, type, or surface of a target. The defender's correct action is logging, not blocking, because false positives would suppress legitimate management traffic that looks similar in flow features.                            |
|  2 |   A   | ACCESS   | **Primary entry attempts.** Direct attempts to obtain unauthorized access to an application, service, or account, including injection-class attacks against the application layer and credential-guessing attacks against the authentication layer. The defender's correct action is throttling: the attempt may be malicious, may be misdirected, or may be a noisy legitimate scanner. |
|  3 |   M   | MANEUVER | **Post-access positioning *or* deployed-payload preparation.** Two distinct phenomena that share the same correct response: (a) lateral movement, persistence, and traffic redirection by a single attacker (MITM, ARP/DNS spoofing); and (b) botnet data-plane preparation by Mirai variants — flooding traffic emitted *prior* to a victim being chosen, used to verify command channels and gather bot health. The defender blocks. |
|  4 |   I   | IMPACT   | **Service degradation against the victim.** Volumetric or protocol DoS/DDoS that directly degrades availability. The defender isolates.                                                                                                                                                                                                                            |

The progression `BENIGN → RECON → ACCESS → MANEUVER → IMPACT` is
**monotone in attacker progress** in this thesis: each step represents
a closer approach to the operational objective (service denial). The
order also induces a natural ordering on the recommended action
(OBSERVE < LOG < THROTTLE < BLOCK < ISOLATE) used by the proportional
reward shape and tested by Phase-3 gate G3 (`docs/results/environment/RESULTS.md`).

## 3 — Label → Stage table (canonical)

The full per-label assignment, frozen for the thesis. The same table
is reproduced for archival completeness in `docs/dataset_card.md` §3;
**this file is the source of truth.**

### Stage 0 — BENIGN
- `BenignTraffic` — only label in the stage; the per-flow reference
  for non-attack behaviour.

### Stage 1 — RECON
- `Recon-PortScan`
- `Recon-OSScan`
- `Recon-HostDiscovery`
- `Recon-PingSweep`
- `VulnerabilityScan` *(held out as OOD; never seen during training)*

### Stage 2 — ACCESS
- `SqlInjection`
- `CommandInjection`
- `XSS` *(held out as OOD)*
- `Backdoor_Malware`
- `BrowserHijacking`
- `Uploading_Attack`
- `DictionaryBruteForce`

### Stage 3 — MANEUVER
- `MITM-ArpSpoofing`
- `DNS_Spoofing`
- `Mirai-greeth_flood`
- `Mirai-greip_flood`
- `Mirai-udpplain` *(held out as OOD)*

### Stage 4 — IMPACT
- `DDoS-ICMP_Flood`
- `DDoS-UDP_Flood`
- `DDoS-TCP_Flood`
- `DDoS-PSHACK_Flood`
- `DDoS-SYN_Flood`
- `DDoS-RSTFINFlood`
- `DDoS-SynonymousIP_Flood`
- `DDoS-ICMP_Fragmentation`
- `DDoS-UDP_Fragmentation`
- `DDoS-ACK_Fragmentation`
- `DDoS-HTTP_Flood` *(held out as OOD)*
- `DDoS-SlowLoris`
- `DoS-UDP_Flood`
- `DoS-TCP_Flood`
- `DoS-SYN_Flood`
- `DoS-HTTP_Flood`

**Total: 34 labels across 5 stages.** Four classes are reserved as
held-out OOD (one per attack stage; see `dataset_card.md` §5 for the
sizing rationale).

## 4 — Per-class rationale for non-trivial assignments

Every assignment in §3 was deliberate. The cases that *could* plausibly
be argued otherwise are addressed below; defending the dissertation
relies on these answers being explicit rather than implicit.

### 4.1 — `MITM-ArpSpoofing` and `DNS_Spoofing` → MANEUVER (not ACCESS)

These attacks do not by themselves grant access to a target; they
**reposition the attacker on the network path** so that traffic that
already exists can be intercepted, redirected, or modified. In MITRE
ATT&CK terms they sit in *Network Sniffing* / *Adversary-in-the-Middle*
under the *Credential Access* tactic, but the act of spoofing a
gateway's ARP entry is a positioning step that precedes either credential
harvesting or downstream session hijacking. Putting them in MANEUVER
captures this *post-arrival, pre-impact* role: the attacker is already
inside the broadcast domain (so RECON is past) but has not yet caused
service degradation (so IMPACT is future). The defender's correct
response — block the offending source — matches our MANEUVER
recommended action.

### 4.2 — `Mirai-greeth_flood`, `Mirai-greip_flood`, `Mirai-udpplain` → MANEUVER (not IMPACT)

This is the most contestable assignment in the table and deserves a
direct answer. CICIoT2023's three `Mirai-*` flood classes are GRE-
encapsulated and UDP-plain flood traffic emitted by Mirai-infected IoT
devices, captured in the testbed *as the bots themselves are being
exercised* — i.e., they are the **bot-side data-plane signature**, not
the *target-side* DDoS that the bots eventually deliver. CICIoT2023's
own paper (Neto et al., 2023) describes these flows as captured during
Mirai bot **operation** rather than during attacks against a specific
external victim.

The reasonable alternative reading — "Mirai produces DDoS, ergo Mirai
is IMPACT" — conflates the *botnet preparation phase* with the
*delivery phase*. From the perspective of an IoT defender on the
infected device, a `Mirai-greeth_flood` flow is not yet damaging *to
the local target*; it is the bot phoning home, exercising its outbound
flooding capability, or participating in distributed reconnaissance of
external targets. The correct local response is to block the bot's
outbound channel (the MANEUVER action), not to declare a service-
denial event (the IMPACT action) which would only be appropriate if
the local device were the *victim* of a flood.

The genuine victim-side IMPACT label set is the 16 `DDoS-*` and
`DoS-*` classes in §3, all of which are captured *at the receiving
endpoint*. Mirai-* labels are captured *at the emitting endpoint*.
The mapping respects this distinction and the result is that the
defender's MANEUVER action — quarantine the locally-infected device
before it joins a broader campaign — is exercised correctly when the
detector sees a `Mirai-*` flow.

### 4.3 — `DictionaryBruteForce` → ACCESS (not RECON)

Brute-force credential guessing is closer to RECON than most ACCESS
attacks (the attacker is, after all, "discovering" the right
credential), and CICIoT2023 places it adjacent to the recon classes in
its own confusion matrix. We map it to ACCESS because the *intent* is
to obtain authenticated entry, not merely to determine whether
authentication is possible: a successful brute-force attempt yields
working credentials, which is the defining outcome of the ACCESS
stage. The defender's correct response (throttle, lock the account,
rate-limit by source) matches the ACCESS recommended action; pure
RECON would be unnecessarily permissive (LOG only).

### 4.4 — `Backdoor_Malware` → ACCESS (not MANEUVER)

`Backdoor_Malware` in CICIoT2023 captures the network signature of a
backdoor being **installed and exercised** on the device, not the
post-installation lateral-movement traffic. The first command-and-
control connection to a freshly-installed backdoor is the moment at
which unauthorized access is achieved — i.e., the ACCESS stage —
even though the backdoor itself is a persistence mechanism. Once
installed, subsequent traffic from the backdoor blends with whatever
secondary action the attacker commands (lateral movement, data
exfiltration, DDoS participation), which would map to MANEUVER or
IMPACT depending on the action; CICIoT2023 does not separately label
these post-install flows, so the `Backdoor_Malware` label refers
unambiguously to the ACCESS-establishing traffic.

### 4.5 — `BrowserHijacking` and `Uploading_Attack` → ACCESS

Both are application-layer entry vectors against a web service running
on the IoT device or its gateway: a hijacked browser session yields
attacker control over an authenticated client, and an upload attack
exploits a file-upload endpoint to deposit attacker-controlled
artefacts. Both fit the ACCESS definition (primary entry attempts at
the application layer) and call for the throttle response.

### 4.6 — `XSS`, `SqlInjection`, `CommandInjection` → ACCESS

The three classical injection classes are the textbook ACCESS case:
the attacker submits crafted input to bypass an application's
authentication, authorization, or sanitization boundary. We do not
distinguish these from credential-based access (`DictionaryBruteForce`)
or persistence-mechanism deployment (`Backdoor_Malware`) because the
defender's correct response is the same for all of them at this level
of abstraction.

### 4.7 — Why all 16 DoS/DDoS classes collapse to IMPACT

CICIoT2023 distinguishes 12 DDoS-* and 4 DoS-* classes by transport
protocol (ICMP, UDP, TCP, HTTP), flag pattern (SYN, RSTFINFlood,
PSHACK), and amplification mode (Synonymous IP, Fragmentation). All 16
share the same operational definition — **active service degradation
against the local victim** — and the same defender response (isolate
the affected service, drop the source, alert). Distinguishing them at
the abstract-stage level would inflate the action space without
changing the optimal policy. Per-class fidelity is preserved at the
detector layer (`src/detector/`) where the 34-label distinction is
visible if a future extension wants to reason at finer granularity.

### 4.8 — Why no `INSTALL` or `C2` stage?

Lockheed Martin's original chain has separate Installation and
Command-and-Control phases. CICIoT2023 does not contain labels that
isolate either phase: backdoor installation traffic is folded into
`Backdoor_Malware`, and there is no labelled C2 channel data. Adding
the stages would create empty cells in every per-stage statistic and
suggest gates that the data cannot adjudicate. The 5-stage abstraction
is intentionally honest about what the dataset can and cannot resolve.

### 4.9 — Why is BENIGN a "stage" rather than a separate axis?

Treating BENIGN as Stage 0 keeps the observation, action, and reward
machinery uniform across attack and non-attack flows. The defender
must still emit an action when the input is benign — the correct one
being OBSERVE — and gating that decision through the same proportional
reward shape that handles the attack stages avoids special-case logic
in the environment (`src/environment/adversarial_env.py`). The
trade-off is that aggregate confusion matrices include a BENIGN row /
column; this is reported transparently in F6
(`docs/results/benchmark/stage_action_proportionality.caption.md`).

## 5 — Properties enforced by the implementation

- **Case-sensitive.** Lookups are byte-exact; `recon-portscan` will
  raise. CICIoT2023 labels are PascalCase and dash-separated; the
  thesis uses them verbatim.
- **Closed.** `AbstractStateLabelMapper.get_stage_id(label)` raises
  `KeyError` on any unmapped label. This is the test-locked invariant
  (`tests/test_label_mapper.py`) that prevents silent drift if a
  future CICIoT release adds a class.
- **Soft alternative.** `get_stage_id_safe(label)` falls back to
  BENIGN with a warning — used **only** during data processing, where
  rejecting an unknown row would abort the entire ingest. The
  training and evaluation paths use the strict variant.
- **Total: 34 labels** mapped, 5 stages used, BENIGN cardinality 1,
  RECON 5, ACCESS 7, MANEUVER 5, IMPACT 16.

## 6 — How to challenge a specific assignment

The committee or any reader who disagrees with a placement above can:

1. Identify the label and the stage they would prefer.
2. Check that moving the label preserves stage monotonicity (the
   relabelled class must remain "less progressed" or "more progressed"
   than the operational definitions allow).
3. Quantify the impact: rerun `python -m scripts.data.derive_stage_labels`
   with the proposed mapping and inspect the resulting per-stage
   confusion matrices. The Phase-2/4/5 results would need to be
   regenerated, which Step 7 of the mentor-review loop is the
   scoped place to do.
4. Update this document, `src/utils/label_mapper.py`,
   `tests/test_label_mapper.py`, and `docs/dataset_card.md` §3
   together.

The four artefacts above are the canonical four-document set for the
mapping. Editing one without the other three is a regression and is
caught by the tests.

## 7 — Source and references

- **Implementation.** `src/utils/label_mapper.py`
  (`KillChainStage` IntEnum, `AbstractStateLabelMapper`).
- **Tests.** `tests/test_label_mapper.py`.
- **Inspiration.** Hutchins, Cloppert, & Amin (2011), *Intelligence-
  Driven Computer Network Defense Informed by Analysis of Adversary
  Campaigns and Intrusion Kill Chains*. The original 7-step Lockheed
  chain. Coarsened here to 5 stages to match what CICIoT2023 actually
  contains.
- **Dataset reference.** Neto, Dadkhah, Ferreira, Zohourian, Lu &
  Ghorbani (2023), *CICIoT2023: A Real-Time Dataset and Benchmark for
  Large-Scale Attacks in IoT Environment*, University of New
  Brunswick. See `docs/papers/` for the full PDF. The Mirai-* class
  semantics in §4.2 derive from §IV-B of that paper.
- **Reward shape that consumes the stages.** `docs/reward-shaping.md`
  (proportional defense reward), `src/environment/adversarial_env.py`
  (recommended-action mapping).
