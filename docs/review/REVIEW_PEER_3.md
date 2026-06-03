# Peer Review: "An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning"

**Reviewer Role:** IEEE/Journal Level-A Reviewer — Specializations: Deep Reinforcement Learning, IoT Security, Cybersecurity Systems
**Document:** Master's Dissertation — UNICAMP / FEEC, 2025
**Author:** Felipe Augusto Oliveira dos Santos
**Advisor:** Prof. Dr. Denis Fantinato
**Review Date:** May 2026
**Review Verdict:** **Major Revision Required**

---

## Executive Summary

This dissertation presents a kill-chain-aware adaptive defense framework for IoT networks combining LSTM-based attack-stage prediction, supervised stage detection, and Deep Reinforcement Learning (DRL). The work demonstrates commendable methodological rigor in its reproducibility protocol, honest disclosure of limitations, and statistical evaluation design. However, several **critical structural flaws** must be addressed before this work reaches the standard expected at a strong venue: (1) a fundamental semantic mismatch between the title's "proactive" claim and the system's reactive deployment behavior; (2) a primary benchmark conducted under a known, reward-misspecified contract; (3) an MDP design where compromise_rate ≡ 1.0 eliminates the primary security metric; and (4) an elevated benign false-positive rate (9.4–12.7%) that disqualifies the system from practical deployment claims. The positive contributions — the reproducibility protocol, honest leakage-bug reporting, and multi-seed bootstrap statistical design — are publication-worthy and should be preserved.

---

## Section 1: Title and Abstract

### 1.1 Title Assessment

**Rating: Needs Revision**

> "An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning"

**Critical Issue — The "Proactive" Claim is Semantically Misleading:**
The title advertises "Proactive Attack Prediction" as a core system property. However, upon reading the methodology, the LSTM stage detector is **not used during RL training** (Section 3.13 explicitly states the `stage_pred` field is "optional, evaluation-time only"). The DRL agents are trained without any stage prediction whatsoever. The LSTM red-team component generates stage-token sequences for episode rolls during training, which is an *environment-generation* tool, not a proactive detection capability deployed by the agent. At inference time, the trained RL agents observe the 29-dimensional feature vector and produce actions — they are **purely reactive**. The only policy that uses a stage predictor is RF-Acting, which is a baseline, not the proposed system.

**Recommended Action:** Revise the title to accurately reflect what is deployed. Options include:
- "A Kill-Chain-Aware Adaptive Defense Framework for IoT Networks via Deep Reinforcement Learning"
- "Autonomous IoT Network Defense via Kill-Chain-Aware Deep Reinforcement Learning and LSTM Threat Modeling"

### 1.2 Abstract Assessment

**Rating: Acceptable with Minor Revisions**

The abstract is informative but contains two issues:

1. **Metric misrepresentation:** Stating PPO "captures 81% of the oracle ceiling" without immediately clarifying that the oracle uses privileged simulator information (true attack stage) unavailable to any deployable system risks misleading readers who skim only the abstract. Add one clause: "(where the oracle uses privileged ground-truth stage information unavailable in deployment)."

2. **Missing key negative result:** The abstract does not mention that all DRL agents are outperformed by RF-Acting in absolute reward, a result that is critical context. The 141× latency advantage should be presented alongside the ~10% reward cost, not separately.

3. **Language consistency:** The abstract (in English) ends abruptly after "Keywords" on a separate page. This is a formatting artifact — the English abstract and keywords should remain on a single page.

---

## Section 2: Document Formatting and Structure

### 2.1 Placeholder Pages — Critical Formatting Error

**Rating: Must Fix Before Submission**

Pages 3 and 4 contain the following placeholder text, verbatim:
- Page 3: *"Inclua aqui o pdf com a ficha catalográfica fornecida pela BAE."*
- Page 4: *"Inclua aqui a folha de assinaturas."*

These are template placeholders that **must be replaced** with the actual library catalog card (ficha catalográfica) and the defense signature page before the dissertation is submitted. This is a submission-blocking error.

### 2.2 Language Inconsistency in Front Matter

The "Agradecimentos" section is written entirely in English. Brazilian university norms (and UNICAMP guidelines) typically require the Agradecimentos to be in Portuguese, with an optional English translation. Verify with the program's formatting guidelines.

### 2.3 Figure Captions: Portuguese vs. English

All figure captions use "Figura N.N –" (Portuguese) and "Tabela N.N –" (Portuguese label) while the body text refers to these as "Figure" and "Table" in English. Pick one language for all cross-references and captions and apply it consistently. If the dissertation is to be in English, captions should read "Figure" and "Table."

### 2.4 "Thesis" vs. "Dissertation" Inconsistency

The document uses both "thesis" (cover page footnote: "Este exemplar corresponde à versão final da **tese**") and "dissertation" (throughout the body) interchangeably. In Brazilian context a master's degree produces a *dissertação* (dissertation); only a doctoral degree produces a *tese* (thesis). The English body should use "dissertation" consistently. Currently both terms appear on the same pages.

### 2.5 Section Numbering Depth

Multiple numbered sections reach four levels deep (e.g., 2.5.0.0.1, 4.5.0.0.2). This is extremely unusual and produces confusing numbering like "2.5.0.0.1." The "0.0" intermediate levels suggest an attempt to add sub-subsections without a proper subsection heading. Recommended fix: restructure to at most three levels of hierarchy and use unnumbered bold paragraph headers (LaTeX `\paragraph{}`) for the fourth level.

### 2.6 Table of Contents — Depth and Readability

The Table of Contents lists entries at the four-level depth, including the "0.0" levels (e.g., "2.5.0.0.1 Deep Q-Networks (DQN)."). This makes the ToC effectively unreadable and non-standard. Limit ToC to three levels maximum.

### 2.7 "SUB" Markers in Body Text

Multiple section headings are followed by the word "SUB" on its own line (e.g., Section 2.2, 2.3, 2.4, etc.). These appear to be LaTeX editor artifacts or draft markers that were never removed. They must be deleted throughout the document.

**Affected locations (non-exhaustive):** After headings in Sections 2.2, 2.3, 2.4, 2.5, 2.7, 2.8, 2.9, 2.10, 3.1, 3.2, 3.6, 3.13, 3.14, 3.21, 3.22, 3.23, 5.5, 5.6.

---

## Section 3: Introduction (Chapter 1)

### 3.1 General Assessment

**Rating: Good with Targeted Revisions**

The introduction effectively motivates IoT security and DRL. The problem statement is clear. However:

### 3.2 Contribution 1 — "Proactive" Architecture Overclaim

Contribution 1 states "A kill-chain-aware proactive-adaptive defense framework" with "an LSTM red-team episode generator." However, the LSTM red-team is an *environment simulator* used during offline training only. The deployed system contains no proactive component. This must be reframed as: "an LSTM-based attack progression simulator used to generate realistic kill-chain training episodes" — fundamentally different from a proactive detection capability.

### 3.3 Contribution 3 — Framing the 141× Latency Advantage

The 141× latency advantage over RF-Acting is real and meaningful, but presenting it as a primary contribution without simultaneously noting that RF-Acting achieves ~10% higher reward is misleading. A reviewer will immediately ask: "if latency is the bottleneck, why not use a faster tree-based classifier or a smaller forest?" The latency-reward tradeoff requires more substantiation, including whether the latency difference is operationally significant in the specific IoT deployment scenarios targeted.

### 3.4 Missing: Quantification of "Proactive" Benefit

If the system claims proactive behavior, there should be a comparison showing what benefit the LSTM-predicted stage adds when consumed by the policy versus not. Currently, the DRL agents train *without* stage predictions and the F15/OOD evaluation does not specifically test whether providing stage predictions to the DRL agents during deployment would close the gap to RF-Acting. This experiment is absent.

---

## Section 4: Background and Related Work (Chapter 2)

### 4.1 General Assessment

**Rating: Adequate but Thin for a Master's Thesis**

### 4.2 Related Work Coverage

The related work table (Table 2.1) covers only **three prior works**. For a master's dissertation in 2025, this is insufficient. Relevant work that should be included:

- DARE (DRL-based Autonomous Response Engine) papers from 2022–2024
- Work on MTD (Moving Target Defense) with RL
- DRL-based NIDS papers (at least 5–8 more recent references from 2022–2025)
- Papers on reward shaping in cybersecurity RL environments
- Papers specifically on Cyber Kill Chain formalization in ML contexts (beyond MAVROEIDIS 2023)
- Work on reproducibility in ML security research (ARES, IEEE S&P)

The current bibliography has 21 references, which is sparse for a master's dissertation. Typical master's dissertations in this domain cite 40–80 works.

### 4.3 Bellman Equations Section (2.4)

Equations 2.1–2.3 are standard textbook content (Sutton & Barto Ch. 3) reproduced verbatim. For a master's dissertation this is acceptable as background, but the treatment adds no value specific to the IoT security context. Consider trimming this to 1–2 paragraphs and instead devoting the space to positioning the chosen MDP formulation against alternatives used in prior cybersecurity RL work.

### 4.4 Oracle Reference Framing (Section 2.6)

The oracle framing is one of the dissertation's stronger conceptual contributions. However, the phrase "not a competing baseline" appears four times in Chapter 2 alone. State it once clearly, then trust the reader. Repetition suggests defensiveness about a result that should instead be contextualized constructively.

---

## Section 5: Methodology (Chapter 3)

### 5.1 General Assessment

**Rating: Significant Issues — Major Revision Required**

### 5.2 CRITICAL — Disconnect Between "Proactive Prediction" and Training

The supervised stage detector (Section 3.11) is explicitly excluded from RL training: "stage_pred (optional, evaluation-time only): the discrete predicted stage from a frozen supervised classifier... the trained DRL agents do not consume this field during training."

This means:
- The DRL agents are trained with NO knowledge of the attack stage
- The "proactive" component is purely decorative during actual RL training
- The system's defense capability is entirely reactive, learned from reward signals, not from explicit stage predictions

**Recommended Action:** Either (a) revise the architecture so the trained DRL agents consume the stage prediction during training and evaluate whether this improves performance, or (b) clearly remove "proactive prediction" from all claims about the DRL agents and relabel the LSTM red-team as an "environment simulator" rather than a "prediction" component. Option (a) would strengthen the contribution; option (b) is the honest correction.

### 5.3 CRITICAL — Red-Team LSTM Generates Tokens, Not Features

The red-team LSTM models stage-token sequences (discrete labels from {benign, recon, access, maneuver, impact}). The feature vectors (29-dimensional per-flow observations) are sampled *independently* from a split-aware pool for whatever stage the LSTM predicts (Section 3.13: "sampled by a split-aware realisation engine").

This creates a fundamental assumption: **temporal independence of feature vectors within an episode**. Consecutive observations in an episode are not drawn from the same attack sequence or flow table — they are independent random samples from a feature pool stratified by stage. Real IoT network attacks produce temporally correlated traffic patterns (e.g., a DDoS attack produces sustained high-volume flows, not random samples from a pool). The disconnect between the LSTM's temporal stage model and the i.i.d. feature sampling undermines the realism of the training environment.

**Recommended Action:** Acknowledge this as a limitation explicitly in Section 3.3 (Monotonic-Attacker Assumption) and in Section 5.3 (Limitations). The current text does not discuss the i.i.d. feature sampling assumption at all.

### 5.4 MDP Design — Compromise Rate ≡ 1.0

Under the primary contract (`impact_is_terminal = True`), the upper-triangular red-team LSTM eventually advances every episode to IMPACT, and the lifecycle-floor clamp prevents premature termination. This means **every episode ends in compromise, regardless of agent behavior** (compromise_rate = 1.0, confirmed in Section 4.9).

This is a fundamental MDP design flaw: if the agent cannot prevent compromise — only mitigate it after the fact — then the system provides no primary security guarantee. The primary metric becomes `mitigated_impact_rate`, but even this is only 29.7% for PPO under the primary contract. The ablation contract (`impact_is_terminal = False`) improves this to 0.900 but still reports compromise_rate = 1.0.

**Recommended Action:** The primary evaluation should use `impact_is_terminal = False` with the structural fix applied. Presenting results under a known-broken primary contract and relegating the fix to an ablation section is backwards. The ablation findings should inform the primary contract design.

### 5.5 Reward Function Complexity and Mis-specification

The reward function (Section 3.15) has **11 scalar parameters** (α, r_prop, p_disp, r_benign, p_over, p_block_benign, p_block_recon, p_impact, b_success, p_missed, p_de_esc). With this many free parameters, the risk of reward mis-specification is high — and it indeed occurs (Section 4.5, G5.4).

**Issues:**
1. The reward mis-specification is discovered post-training. The systematic approach should have been to formally verify calibration properties (which the 13 calibration tests partially do) before training, and then test empirically whether the trained policy matches the intended behavior before reporting it as the primary result.
2. The 13 calibration tests verify the *expected policy behavior* under the reward, not whether RL training will converge to that behavior. These are different things.
3. The de-escalation bonus (+250) dominates all other terms. A trained agent earning ~6.3 de-escalations/episode at +250 each = +1575 per episode from this single component dwarfs the impact penalty (-200) and the missed-impact penalty (-150). This imbalance should have been caught during the calibration test design phase.

**Recommended Action:** Redesign the reward function with the `impact_is_terminal = False` fix as the primary contract. Add a de-escalation rate cap or diminishing returns to the de-escalation bonus to prevent farming.

### 5.6 Episode Lifecycle — Lifecycle-Floor Impact Clamp

Section 3.14 introduces a "lifecycle-floor impact clamp" that downgrades impact transitions before step 20 to maneuver. This is an ad-hoc engineering patch that:
- Biases MTTC values toward exactly min_episode_length = 20
- Makes the MTTC metric essentially uninformative (confirmed in Section 4.9)
- Is not derived from any empirical analysis of real IoT attack progression speeds

The clamp should either be removed and replaced with a more principled minimum episode length constraint, or its impact on results should be quantified (what fraction of episodes are clamped? How does removal affect results?).

### 5.7 Defender Observability — Missing Feature Description

Section 3.4 states the defender observes "a 29-dimensional per-flow feature vector derived from the CICIoT2023 traffic capture" but never lists the 29 features. This is a significant omission. Reviewers and practitioners need to know what network observables are used. A table listing all 29 features (or at minimum feature categories) should be in Appendix B alongside the kill-chain mapping.

### 5.8 Splits Protocol — n=150 vs n=300 Evaluation Asymmetry

Section 3.20 states n=300 episodes for DRL agents and n=150 for baselines and oracle. This asymmetry is unexplained and potentially introduces a bias in the bootstrap CI width comparison. All policies should be evaluated on the same number of episodes for a fair comparison.

### 5.9 Blue-Team Training — 250k vs 500k Steps Inconsistency

Section 3.19 states training for 250,000 steps. Figure 4.4 (learning curves) appears to show x-axis extending to 500,000 timesteps. This needs verification — either the training was done for 500k steps (contradicting the text) or the figure x-axis range is incorrect.

---

## Section 6: Results and Discussion (Chapter 4)

### 6.1 General Assessment

**Rating: Good Statistical Rigor, Significant Framing Issues**

### 6.2 CRITICAL — Primary Results Under Mis-specified Reward

The headline results in Section 4.6 (81% oracle capture, 141× latency advantage, Table 4.3) are all obtained under `impact_is_terminal = True`, which the dissertation itself identifies as a reward-misspecified contract in Section 4.5. This means the primary benchmark showcases an agent optimizing a proxy objective that "diverges from the human's intended objective" (Section 4.5).

Reporting the ablation fix (Section 4.7, `impact_is_terminal = False`, mitigated-impact rate 0.900) as secondary to the broken-contract results inverts the logical priority. A reviewer will reasonably ask: "Why are the primary results under a known-bad contract?"

**Recommended Action:** Restructure so the `impact_is_terminal = False` contract is the primary benchmark. The `True` contract results can be presented as an ablation or historical baseline showing the improvement.

### 6.3 RF-Acting Outperforms All DRL Agents

Table 4.3 shows RF-Acting (+1486.2) significantly outperforms PPO (+1334.5), A2C (+1296.7), and DQN (+1218.9). The statistical separation is confirmed in Table 4.4 (Cohen's d = -0.755 for DQN vs. RF-Acting, p < 0.0001).

The dissertation frames this as a "latency-reward trade-off frontier" rather than RF-Acting dominating DRL. While the framing is defensible (141× latency advantage), it raises the question: **is the DRL contribution worthwhile?** The dissertation should more directly address this. Questions a reviewer will raise:

1. Could a smaller RandomForest (e.g., 10 trees instead of 100) close the latency gap while maintaining most of the reward advantage?
2. Could the RF-Acting composite be further optimized (e.g., using the MLP detector instead of RF for faster inference)?
3. At what reward cost would RF-Acting match the RL latency (e.g., by caching predictions)?

Without these comparisons, the 141× latency advantage vs. RF-Acting is cherry-picked against an unnecessarily slow baseline configuration.

### 6.4 Benign False-Positive Rate — Deployment-Blocking Limitation

Section 4.6 reports benign-FPR of 9.4% (A2C), 9.6% (PPO), and 12.7% (DQN). The dissertation notes this "substantially exceeds the 1% operational threshold."

For context: a 9.6% false-positive rate on benign traffic means the system triggers a BLOCK or ISOLATE action approximately 1 in 10 times on legitimate traffic. In an IoT environment with thousands of benign transactions per minute, this would be operationally catastrophic. This is **not a minor limitation** — it fundamentally disqualifies the current system from practical deployment and should be positioned as a central unsolved problem, not buried in Section 4.6 as a paragraph.

**Recommended Action:** Add a dedicated limitation section for benign-FPR. Quantify the operational impact (e.g., "at 1000 decisions/minute, PPO would incorrectly block ~96 legitimate flows per minute"). This reframing would sharpen the future work contribution (reward redesign with explicit FPR constraint, Direction 6).

### 6.5 MTTC as a Metric — Uninformative Under Current Design

Mean Time to Compromise (MTTC) is reported throughout (Table 4.3, Figure 4.9) with values clustered around 19.1–19.4 steps for all policies including Random. The lifecycle-floor impact clamp forces MTTC toward min_episode_length=20 regardless of policy quality.

**Consequence:** MTTC provides essentially zero discriminative power between policies and should either be removed from the primary results tables or accompanied by a clear caveat in the table caption. Currently it appears alongside meaningful metrics without sufficient warning.

### 6.6 Statistical Tests — Missing Key Comparison

Table 4.4 reports pairwise tests for DQN vs. PPO, DQN vs. A2C, and DQN vs. RF-Acting. Conspicuously absent:
- **PPO vs. RF-Acting** (the most policy-relevant comparison)
- **A2C vs. RF-Acting**
- **PPO vs. A2C**

These are the tests that would establish the statistical significance of the key claims. Their absence is notable.

### 6.7 Figure 4.4 — Learning Curves x-Axis Discrepancy

The figure description mentions "episodic reward over training for DQN, PPO, A2C across ten seeds each" with training at 250,000 steps, but the figure's x-axis appears to extend to 500,000 timesteps. If this is simply a figure generated from a 500k sweep with 250k being the effective training horizon, the x-axis should be clipped or annotated. If training actually ran to 500k, the methodology text must be corrected.

### 6.8 Figure 4.9 — Security Metrics Table as Figure

Figure 4.9 presents the security metrics as a rendered table-as-figure. This is non-standard and makes the data inaccessible for citation. The content of Figure 4.9 is already captured in Table 4.3 and Table 4.5. Recommend merging the information into properly formatted LaTeX tables and removing Figure 4.9.

### 6.9 OOD Robustness — VulnerabilityScan Explanation Circular

Section 4.8 explains that RF-Acting outperforms DRL on VulnerabilityScan because "RF mostly predicts benign when shown a VulnerabilityScan row, and the recommended action for benign is observe (zero defensive force, no disproportionate-action cost)." This means RF-Acting succeeds by *failing gracefully* — its mis-predictions happen to be cost-free.

This observation, while honest, undermines the RF-Acting baseline's validity as a meaningful comparison point for OOD robustness. A truly robust policy should correctly identify novel recon-stage attacks, not accidentally avoid penalties through mis-classification. The section should discuss this more critically rather than presenting it as "evidence that the RandomForest is doing useful work."

### 6.10 Figure 4.5 — Action Distribution Evolution for PPO Only

The action-distribution evolution is shown only for PPO. For completeness and to support the statistical separation claims, equivalent figures for DQN and A2C should be included (possibly in an appendix). The training behavior differences between the three algorithms are currently described textually but not visualized for DQN and A2C.

---

## Section 7: Conclusions and Future Work (Chapter 5)

### 7.1 General Assessment

**Rating: Good Structure, Overclaims in Summary**

### 7.2 Contribution 3 Overclaim in Summary

Section 5.1 Contribution 3 states: "The trained agents capture 81% of the oracle ceiling... with non-overlapping bootstrap confidence intervals against every non-RL trivial baseline."

The phrase "every non-RL trivial baseline" excludes RF-Acting from this claim. However, RF-Acting IS the most operationally relevant deployable baseline, and DRL does NOT beat it in reward. The sentence structure implies DRL dominates all deployable alternatives, which is false. Recommend: "with non-overlapping bootstrap confidence intervals against all trivial baselines (random, always-observe, always-block), though RF-Acting achieves higher reward at substantially higher latency."

### 7.3 "Findings Worth Defending" — Framing Issues

Section 5.2 presents three findings as "worth defending." The term "worth defending" is non-standard academic language and suggests defensiveness. Rename to "Principal Empirical Findings" or "Key Results."

Finding 2 contains: "the original 'RL > recommended-action rule' threshold reads passes: false permanently." This implementation detail about a JSON scoreboard field has no place in a Conclusions chapter.

### 7.4 Future Work Directions — Insufficiently Motivated

Directions 4 (MARL), 5 (Federated Learning), and 6 (FPR constraint) are reasonable but described too briefly (1–2 paragraphs each). For a master's thesis, future work should include preliminary motivation (why is this the *next* step, what specific challenge does it address) and ideally a preliminary feasibility assessment.

Direction 3 (train-time OOD augmentation) is the most directly motivated by empirical findings and should be the first direction listed, not the third.

---

## Section 8: Reproducibility Protocol (Appendix A)

### 8.1 General Assessment

**Rating: Excellent — Publication-Worthy Contribution**

The SHA-256 hash chain protocol, smoke-reproducibility harness, and manifest.json discipline are among the strongest aspects of this work. The 458 OK / 0 FAIL verdict with 2 documented KNOWN-DIVERGENCE entries and 6 SKIP entries is exemplary. This protocol exceeds the reproducibility standards of most published ML security papers and should be highlighted as a standalone contribution.

### 8.2 Minor Issues

1. The verification recipe in Section A.4 references `make thesis-image` (Docker build). The Dockerfile and its dependencies should be included in the repository and documented. A reviewer attempting reproduction should not need to infer Docker environment requirements.

2. The `KNOWN-DIVERGENCE` table is described but its contents are not reproduced in the appendix. For a dissertation, the two known divergences (pre/post leakage-fix splits SHA) should be enumerated inline.

---

## Section 9: Appendices Assessment

### 9.1 Appendix B — Kill-Chain Mapping (Table B.1)

The mapping is sound and the design rationale reference to `docs/kill-chain-mapping.md` is appropriate. However:

- The rationale for mapping Mirai-greeth_flood and Mirai-greip_flood to MANEUVER (rather than IMPACT) should be explained in the body text. These are volumetric attack variants typically considered impact-stage. The mapping decision is consequential for training and deserves justification in Section 3.8, not just in an external document.
- Classes marked ⋆ (OOD) are noted but the selection criteria for which 4 of 34 classes become OOD is not stated. Was this random? Stratified by stage? The selection methodology should be documented.

### 9.2 Appendix C — Hyperparameters

Table C.1 is well-structured. One issue: the note says "Values are Stable Baselines3 defaults" but the dissertation should justify why defaults are used rather than tuned hyperparameters. For PPO especially, the batch_size=64 and n_epochs=10 are SB3 defaults; a hyperparameter search could substantially change results.

### 9.3 Missing Appendix — Feature Descriptions

As noted in Section 5.7 of this review, the 29 CICIoT2023 features used in the 29-dimensional per-flow vector are never listed. Add an appendix listing all features, their types, and any preprocessing applied (normalization, standardization). The dissertation mentions a "scaler" in the reproducibility manifest; its type and parameters should be documented.

---

## Section 10: Technical and Scientific Issues Summary

### 10.1 Must Fix (Blocking Issues)

| # | Issue | Location | Severity |
|---|-------|----------|----------|
| M1 | Placeholder pages (ficha catalográfica, signatures) | Pages 3–4 | BLOCKING |
| M2 | "Proactive" claim misrepresents system architecture | Title, Abstract, Ch.1, Ch.3 | CRITICAL |
| M3 | "SUB" markers throughout body text | Throughout | BLOCKING |
| M4 | Primary results under known reward-misspecified contract | Ch. 4 | CRITICAL |
| M5 | compromise_rate ≡ 1.0 makes primary security metric degenerate | Ch. 3, 4 | CRITICAL |
| M6 | No feature list for 29-dim observation vector | Ch. 3, Appendix | MAJOR |
| M7 | n=300 vs n=150 episode asymmetry in benchmark | Sec 3.20, 4.6 | MAJOR |

### 10.2 Should Fix (Significant Issues)

| # | Issue | Location | Severity |
|---|-------|----------|----------|
| S1 | Benign FPR 9.4–12.7% understated as minor limitation | Sec 4.6, 5.3 | SIGNIFICANT |
| S2 | MTTC metric uninformative, should be removed or annotated | Tables 4.3, 4.5 | SIGNIFICANT |
| S3 | Missing pairwise tests: PPO vs RF-Acting, A2C vs RF-Acting | Table 4.4 | SIGNIFICANT |
| S4 | RF-Acting latency not minimized (100-tree forest vs smaller alternatives) | Sec 4.6 | SIGNIFICANT |
| S5 | Section numbering uses "0.0" levels (2.5.0.0.1 etc.) | Throughout | SIGNIFICANT |
| S6 | Figure 4.9 (table-as-figure) should be a LaTeX table | Ch. 4 | SIGNIFICANT |
| S7 | Learning curves x-axis discrepancy (250k vs 500k) | Figure 4.4 | SIGNIFICANT |
| S8 | Related work covers only 3 prior works | Ch. 2 | SIGNIFICANT |
| S9 | i.i.d. feature sampling assumption not disclosed | Sec 3.10, 5.3 | SIGNIFICANT |
| S10 | Mirai-greeth/greip MANEUVER mapping unjustified | Appendix B | MODERATE |

### 10.3 Should Consider (Improvement Opportunities)

| # | Issue | Location |
|---|-------|----------|
| C1 | DRL agents consuming stage_pred during training (missing experiment) | Sec 3.13 |
| C2 | Smaller RF forest for latency-reward tradeoff analysis | Sec 4.6 |
| C3 | Action distribution figures for DQN and A2C | Ch. 4 |
| C4 | Formal FPR analysis with operational impact quantification | Sec 4.6, 5.3 |
| C5 | De-escalation bonus farming analysis depth | Sec 4.5 |
| C6 | Pre/post leakage-fix performance comparison | Sec 4.2 |
| C7 | OOD class selection criteria documentation | Appendix B |
| C8 | Hyperparameter search justification or sensitivity | Appendix C |

---

## Section 11: Writing Quality Assessment

**Rating: Good — Some Structural Revision Needed**

### 11.1 Positive Aspects
- Technical terminology is used correctly throughout
- Limitation disclosure is honest and appropriately detailed
- The pre-registered finding methodology adds scientific rigor
- Bootstrap CI reporting is consistent and correct
- The audit-first framing is well-communicated

### 11.2 Issues

**Defensive Repetition:** The phrase "the oracle reads the simulator's privileged stage information and is a measurement instrument, not a competing baseline" appears in nearly identical form at least 6 times across Chapters 1–5. State this clearly once in Chapter 2 and thereafter use a shorter reference ("the oracle (Section 2.6)").

**Excessive Hedge Language:** Phrases like "the right question... is not X but Y" (Section 4.6) and "Findings Worth Defending" (Section 5.2) read as responses to anticipated criticism rather than confident scientific communication. Restructure these as straightforward positive claims supported by evidence.

**Section 4.5 — Reward Mis-specification Discovery:** The phrase "This is a textbook case of reward mis-specification" is accurate but should cite a specific reference (e.g., Krakovna et al. 2020, "Avoiding Side Effects in Complex Environments" or Leike et al. 2018, "AI Safety Gridworlds") rather than just Sutton & Barto, which does not specifically address reward mis-specification.

**Gate IDs in Body Text:** References to internal development gate IDs (G5.4 PASS-WITH-FINDING, G6.3 PASS, etc.) appear throughout the Results chapter. While the internal development protocol is meritorious, these gate references are not meaningful to external readers. Either move them entirely to Appendix A or provide a brief legend when they first appear.

---

## Section 12: What Should Be Re-Run / Re-Done / Removed

### 12.1 Experiments to Re-Run

| Experiment | Reason | Priority |
|-----------|--------|----------|
| Primary benchmark using `impact_is_terminal=False` | Current primary uses mis-specified contract | HIGH |
| All DRL training with stage_pred consumed at training time | Tests "proactive" claim empirically | HIGH |
| Benchmark with equal n episodes across all policies (n=300) | Current asymmetry biases CI widths | MEDIUM |
| RF-Acting with 10-tree and 50-tree forests | Tests whether 100-tree latency is necessary | MEDIUM |
| Pairwise statistical tests: PPO vs RF-Acting, A2C vs RF-Acting | Missing key comparisons | MEDIUM |

### 12.2 Content to Remove or Substantially Revise

| Content | Action | Reason |
|---------|--------|--------|
| "Proactive" in title and all major claims | Revise or remove | Semantically incorrect for deployed system |
| Figure 4.9 (table-as-figure) | Convert to LaTeX table or remove | Non-standard, data already in tables |
| MTTC metric in primary tables | Remove or annotate heavily | Uninformative under lifecycle-floor clamp |
| "SUB" markers | Delete | Editorial artifacts |
| Placeholder pages 3–4 | Replace with actual content | Missing required elements |
| Findings 2 JSON scoreboard reference | Remove from Ch. 5 | Implementation detail in Conclusions |

### 12.3 Content to Add

| Content | Location | Priority |
|---------|----------|----------|
| 29-feature list | New Appendix D or within Appendix B | HIGH |
| i.i.d. feature sampling limitation disclosure | Sec 3.3 / 5.3 | HIGH |
| PPO vs RF-Acting statistical test | Table 4.4 | HIGH |
| OOD class selection criteria | Appendix B / Sec 3.9 | MEDIUM |
| FPR operational impact quantification | Sec 4.6 / 5.3 | MEDIUM |
| 15–25 additional related work citations | Ch. 2 | MEDIUM |
| Smaller RF forest latency-reward analysis | Sec 4.6 | MEDIUM |
| Action distribution figures for DQN, A2C | Appendix or Ch. 4 | LOW |

---

## Section 13: Overall Scoring

| Dimension | Score (1–5) | Comments |
|-----------|-------------|----------|
| Originality | 3/5 | Framework combination is novel; individual components are standard |
| Technical Rigor | 3/5 | Strong statistics; MDP design flaw undermines primary results |
| Reproducibility | 5/5 | Exceptional — publication-standard protocol |
| Writing Clarity | 3/5 | Good but repetitive defensiveness; formatting issues |
| Experimental Design | 2/5 | Benchmark under mis-specified contract; missing key comparisons |
| Related Work Depth | 2/5 | Only 3 prior works compared; 21 total references too sparse |
| Contribution Clarity | 2/5 | "Proactive" title misrepresents deployed system |
| Formatting | 2/5 | Placeholder pages, SUB markers, numbering issues |

**Overall Rating: 2.75/5 — Major Revision Required**

---

## Section 14: Summary Recommendations for Revision Agent

The following ordered action list is provided for automated planning of revisions:

### Priority 1 — Must Complete Before Resubmission

1. **Replace placeholder pages** (pages 3–4) with ficha catalográfica and signature page.
2. **Delete all "SUB" markers** from section headings throughout the document.
3. **Revise title** to remove "Proactive Attack Prediction" or reframe to accurately reflect system architecture.
4. **Restructure primary benchmark** to use `impact_is_terminal=False` as the primary contract and demote `impact_is_terminal=True` results to ablation context.
5. **Add 29-feature list** as a new appendix or within Appendix B.
6. **Equalize evaluation episodes** to n=300 for all policies (re-run baselines and oracle with 300 episodes).
7. **Fix section numbering** — eliminate "0.0" intermediate levels (2.5.0.0.1 → bold paragraph headers).

### Priority 2 — Strong Improvement

8. **Add PPO vs RF-Acting and A2C vs RF-Acting statistical tests** to Table 4.4.
9. **Disclose i.i.d. feature sampling assumption** in Section 3.3 and Section 5.3.
10. **Elevate benign-FPR** to a central limitation with operational impact quantification.
11. **Remove or heavily annotate MTTC** from primary results tables.
12. **Convert Figure 4.9** to a proper LaTeX table.
13. **Expand related work** to include 15–25 additional references.
14. **Reduce oracle-framing repetition** to a single clear statement in Section 2.6.
15. **Verify Figure 4.4 x-axis** — confirm whether training ran to 250k or 500k steps and align figure and text.

### Priority 3 — Recommended Enhancements

16. Run ablation: **DRL agents consuming stage_pred during training** to empirically test proactive-prediction benefit.
17. Run sensitivity: **RF-Acting with smaller forests** (10-tree, 50-tree) for latency-reward curve.
18. Add **action distribution visualizations for DQN and A2C** in appendix.
19. Document **OOD class selection criteria** in Section 3.9.
20. Add **reward mis-specification citations** beyond Sutton & Barto (e.g., Krakovna et al. 2020).
21. Standardize language: **"dissertation" throughout** (remove "thesis" in English body); **"Figure/Table"** in captions (not "Figura/Tabela").
22. Move **gate ID references** (G5.4, G6.3, etc.) to Appendix A or provide a legend on first use.
23. Justify **OOD class Mirai-greeth/greip MANEUVER mapping** in Section 3.8 text.

---

*End of Review*
*This review document is intended for consumption by a revision-planning agent. All section references refer to the submitted dissertation (principal.pdf). Recommendations marked Priority 1 are submission-blocking; Priority 2 are required for a strong revision; Priority 3 improve quality and strengthen contributions.*
