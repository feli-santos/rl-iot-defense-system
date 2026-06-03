# Peer Review: *An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning*

**Review Type:** IEEE / Journal-level technical review
**Reviewer Expertise:** Reinforcement Learning, IoT Security, Cyber-physical Systems, Empirical ML
**Document Version:** Master’s Dissertation, UNICAMP/FEEC, 2025
**Author:** Felipe Augusto Oliveira dos Santos
**Advisor:** Prof. Dr. Denis Fantinato

---

## 1. Overall Assessment

This dissertation presents a kill-chain-aware adaptive defense framework for IoT networks that combines an LSTM-based red-team episode generator, a supervised stage detector, and Deep Reinforcement Learning (DQN, PPO, A2C) evaluated on the CICIoT2023 dataset. The work is methodologically ambitious, emphasizes reproducibility via SHA-256 hash chains, and honestly reports a discovered reward mis-specification. These are commendable strengths.

However, **the manuscript in its current form is not ready for submission to a top-tier IEEE security or ML venue** (e.g., *IEEE TDSC*, *TIFS*, *TNSM*, *ACSAC*, *AAAI*). The primary experimental contract (`impact_is_terminal=True`) produces policies that fail at the core security objective (mitigating impact), yet the abstract and main benchmark present these as headline results. The benign false-positive rate (9–13%) is operationally disqualifying and is treated as a minor limitation rather than a critical flaw. Several key ablations are missing (e.g., RL trained with detector input, RF-Acting with smaller forests, non-monotonic attacker stress-tests). The narrative frequently conflates internal audit artifacts (gate scoreboards, pass/fail verdicts) with scientific exposition, which distracts from the empirical argument.

**Bottom line:** With a *major revision* that re-centers the experimental narrative on the structurally fixed reward contract, adds missing ablations, and reframes the security claims more conservatively, this work could become a strong journal submission. As it stands, the thesis over-claims deployability and under-reports security-critical failure modes.

---

## 2. Critical Issues (Must Be Addressed Before Any Publication)

### 2.1 The Primary Experimental Contract Invalidates the Core Security Claim
**Severity: Critical**

The default environment contract (`impact_is_terminal=True`) allows the red-team LSTM to transition to *impact* and terminate the episode before the defender can act at the impact stage itself. Under this contract, the trained PPO agent achieves a **mitigated-impact rate of only 0.297** (Table 4.5, p. 57), meaning it fails to defend against impact ~70% of the time. The agent “farms” de-escalation bonuses earlier in the episode and accepts the terminal impact loss—a classic reward-mis-specification (Section 4.5.0.0.2, p. 35).

**Problem:** The abstract (p. 7), introduction (p. 20), and held-out benchmark (Section 4.6, p. 37) all present the **+1334.5 reward / 81% oracle capture** as the headline result. Yet this policy is structurally unable to perform the stated mission of the system. A defense system that allows 70% of attacks to reach impact is not “adaptive defense”; it is a reward-optimized de-escalation collector.

**Required Action:**
- **Pivot the primary benchmark to the fixed contract** (`impact_is_terminal=False`), which raises the mitigated-impact rate to **0.900** (Section 4.7, p. 59). The abstract, introduction, and all headline figures must report the **+1542 reward / ~94% oracle capture** (1542/1647.6) as the main result.
- The default contract should be demoted to a *“reward-mis-specification case study”* (Section 4.5.0.0.2), not the primary evaluation. If the default is retained as a primary result, the paper must be reframed as a *study of reward hacking in security RL*, not as a deployable defense system.

### 2.2 Benign False-Positive Rate Is Operationally Unacceptable
**Severity: Critical**

All trained DRL agents exhibit a benign-stage false-positive rate of **9.4–12.7%** (Section 4.6.0.0.5, p. 41). In network security, a >1% benign FPR is generally considered operationally untenable because it causes massive service disruption (denial-of-service by the defender itself). The thesis notes this as a “limitation requiring future reward-redesign work,” but this is insufficient.

**Required Action:**
- Elevate the FPR to a **primary threat-to-validity** in the abstract and conclusions.
- Add a **hard FPR ablation**: e.g., add a Lagrangian penalty or a constrained MDP formulation that caps benign FPR at 1% and report the reward cost. If this is out of scope for the thesis, state explicitly that the system is **not deployable** until this is resolved.
- Remove or heavily qualify language such as “deployable frontier” (p. 41) and “deployable baseline” when referring to the trained RL agents. A 10% benign-FPR policy is not deployable.

### 2.3 The “Oracle” Is Not Clairvoyant About Future Stage Transitions
**Severity: Major**

The Recommended-Action oracle reads the *current* true stage, not the *next* stage (Section 2.6, p. 26). Because the red-team LSTM permits **stage-skipping** (e.g., benign → impact) under the upper-triangular mask, the oracle can be surprised by an impact transition while it is still recommending *observe* for the benign stage. This explains the seemingly impossible **oracle mitigated-impact rate of 0.267** (Table 4.5, p. 57).

**Required Action:**
- Explicitly clarify in Section 2.6 and 3.12 that the oracle is a **current-stage oracle**, not a next-stage predictor.
- Discuss whether a *next-stage* oracle (using the red-team LSTM’s transition distribution) would provide a tighter upper bound, and why the current oracle is still a valid baseline for the stage-detection task.
- If stage-skipping is not intended to be realistic, constrain the LSTM to single-stage advances (benign→recon→access→maneuver→impact) and re-run the oracle baseline. The current setup conflates stage-skipping (a modeling choice) with oracle performance.

### 2.4 Missing Ablations That Are Essential for the Claim
**Severity: Major**

Several ablations are standard in RL-for-security papers and are conspicuously absent:

1. **RL trained with stage-detector input:** The trained DRL agents do not consume the `stage_pred` feature during training (Section 3.13, p. 21). An ablation showing PPO trained *with* the MLP stage prediction would quantify how much of the 19% oracle gap is due to hidden-state uncertainty versus RL optimization failure. This is a single re-run and would add enormous value.
2. **RF-Acting with smaller forests:** The 141× latency claim compares PPO to a 100-tree RandomForest. A 10- or 20-tree forest would likely close much of the reward gap while reducing latency by an order of magnitude. Without this sweep, the latency–reward trade-off is straw-manned.
3. **Non-monotonic attacker stress test:** The monotonic-attacker assumption (Section 3.3, p. 15) is extremely restrictive. Even a single experiment with a small retreat probability (e.g., 5%) would reveal whether the defender policy collapses when the MDP violates its inductive bias.

**Required Action:** Add at least ablations (1) and (2) to the main benchmark. Ablations should be run with the fixed contract (`impact_is_terminal=False`).

---

## 3. Major Comments (Significant Improvements Needed)

### 3.1 Abstract and Introduction Are Misleading
The abstract (p. 7) states the system is evaluated on a “five-stage Cyber Kill Chain” and reports the 81% capture. It does **not** mention that the default contract yields a 29.7% impact-mitigation rate or a 10% benign FPR. An abstract must contain the most important caveat that prevents misinterpretation.

**Required Action:** Rewrite the abstract to:
1. Report the fixed-contract result (~94% oracle, 90% mitigated impact) as primary.
2. State the benign FPR explicitly (~10%) and flag it as a deployability blocker.
3. Remove the phrase “all results are reproducible by a chain of hashes SHA-256 verified of point a” — this is garbled and sounds unprofessional. Replace with “All results are reproducible via a SHA-256-verified artifact chain.”

### 3.2 The “Leakage Bug” Should Not Be a Primary Contribution
The discovery and fix of the OOD data leakage (Section 3.9, p. 18; Section 4.2, p. 28) is presented as Contribution #2 (p. 20). While methodological hygiene is laudable, **catching one’s own bug is not a scientific contribution** suitable for an abstract. It belongs in a reproducibility or methodology appendix.

**Required Action:** Condense the leakage narrative in the introduction to one sentence. Move the detailed forensics to Appendix A or a dedicated “Reproducibility & Data Hygiene” section. Replace Contribution #2 with something substantive, e.g., the reward-function structural analysis or the latency–reward Pareto characterization.

### 3.3 OOD Robustness Claims Are Over-Spun
Section 4.8 (p. 45) frames the OOD result as “RL is robust to, not better at, the hardest OOD class.” On VulnerabilityScan, trained PPO (+1313) is outperformed by RF-Acting (+1611). The explanation—that RF-Acting is “wrong in a cheap way” (p. 46)—is technically correct but does not change the empirical fact that the **static baseline beats the adaptive RL agent on a novel attack**. Calling this “robustness” is generous; “failure to generalize beyond in-distribution feature statistics” is more accurate.

**Required Action:** Reframe Section 4.8 as a **limitation**, not a finding worth defending. State clearly: *“The trained RL agent does not outperform the static baseline on feature-novel attacks, indicating that representation learning from CICIoT2023 traffic features does not automatically yield OOD generalization.”*

### 3.4 Audit Protocol Artifacts Clutter the Main Text
The main text repeatedly uses internal audit language: “gate G5.4 PASS-WITH-FINDING,” “audit-AF2 reframe,” “D7.9.1 finding-activation” (pp. 35, 37, 45, 50). This is project-management metadata, not scientific exposition. It breaks the narrative flow and will confuse journal readers.

**Required Action:** Remove all gate codes (e.g., G6.2, G7.2) and audit verdicts from the main text. Retain them only in Appendix A. In the main text, describe findings in standard scientific language: *“We discovered a reward mis-specification (Section 4.5) and fixed it structurally (Section 4.7).”*

### 3.5 Related Work Positioning Needs Sharpening
Table 2.1 (p. 12) compares this work to IoTWarden, HoneyIoT, and Nguyen et al. The comparison is fair but qualitative. The thesis correctly notes that direct numerical comparison is impossible due to different environments (p. 28). However, it misses an opportunity to compare **algorithmic choices** (e.g., why PPO vs. the DQN used in IoTWarden? Why no self-play or federated learning baselines here?).

**Required Action:** Add a paragraph in Section 2.11 discussing why prior algorithmic choices (DQN-only in IoTWarden, A3C in HoneyIoT) might have been suboptimal, and how the multi-algorithm sweep in this thesis adds evidence to the RL-for-security literature.

---

## 4. Detailed Section-by-Section Feedback

### 4.1 Title
**Current:** *An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning*
**Assessment:** Accurate but generic. It does not convey the key differentiators (kill-chain framing, CICIoT2023 grounding, reproducibility protocol).
**Suggestion:** Consider a more specific title, e.g., *“Kill-Chain-Aware Deep Reinforcement Learning for IoT Intrusion Response: A Reproducible Benchmark on CICIoT2023.”* (Optional; current is acceptable if abstract is fixed.)

### 4.2 Abstract (p. 7)
- **Line 1-2:** “exponential proliferation... has produced a vast network” — cliché. Start with the concrete problem: static IDS fails on IoT due to resource constraints and non-stationarity.
- **Line 6-7:** “captures 81% of the oracle ceiling” — **must be updated** to the fixed-contract result (~94%) or clearly caveated.
- **Last line:** “all results are reproducible by a chain of hashes SHA-256 verified of point a” — **nonsense phrase**. Rewrite entirely.

### 4.3 Introduction (Chapter 1)
- **Section 1.1 (p. 18):** The citation of Vailshery (2025) and Morgan (2025) for IoT growth and cybercrime damages is fine, but these are industry reports. Add a peer-reviewed survey (e.g., Mavroeidis, 2023 is already cited) to anchor the threat model academically.
- **Section 1.2 (p. 19-20):** The five contributions are listed. As noted, Contribution #2 (leakage bug) should be demoted. Contribution #3 (141× latency) should be caveated with the benign-FPR issue.
- **Page 20, last paragraph:** “The remainder of this dissertation is structured as follows.” Standard, but ensure it matches the actual chapter structure (it does).

### 4.4 Background and Related Work (Chapter 2)
- **Section 2.1 (p. 22-23):** The five-stage kill chain is well-explained. However, the mapping from MITRE ATT&CK to the five stages is asserted but not justified in detail. Add a paragraph explaining why *Credential Access* maps to *access* rather than *maneuver*, as this will be scrutinized by security reviewers.
- **Section 2.5 (p. 24-26):** The DQN, PPO, and A2C descriptions are textbook. Add one sentence per algorithm explaining why it is suitable for a discrete-action, high-frequency network defense task.
- **Section 2.6 (p. 26):** The oracle framing is excellent and methodologically mature. Expand slightly to address the stage-skipping issue noted in Section 2.3 of this review.
- **Section 2.10 / Table 2.1 (p. 12):** The table is useful but the “Deployable?” column for HoneyIoT says “No (no reward comparison)”, which is a confusing concatenation of two issues. Split into two columns: “Deployable?” and “Benchmarked against non-RL baseline?”.

### 4.5 Methodology (Chapter 3)
- **Section 3.3 (p. 15):** Monotonic-attacker assumption. Add a formal definition (e.g., “∀t, s_{t+1} ≥ s_t”) and a diagram showing allowed vs. forbidden transitions. This is central to the MDP.
- **Section 3.5 / Figure 3.1 (p. 16):** The architecture figure is clear. Ensure the font size is readable when printed. The dashed arrow labeled “eval-time” is good. Add a small note in the caption that the Stage Detector is *not* used during RL training (only at eval).
- **Section 3.8 (p. 18):** Kill-chain projection. The mapping table is in Appendix B. Summarize the heuristic in one paragraph in the main text so the reader does not have to flip back and forth.
- **Section 3.9 (p. 18):** OOD reservation. The leakage bug narrative is too long. Condense to: *“During development, we discovered that OOD indices were not excluded from the stratified split. The fix (commit 3cd2fb9) and disjointness assertions are detailed in Appendix A.”*
- **Section 3.10 (p. 19):** Red-team LSTM. Why a single-layer LSTM? Justify the architecture choice (e.g., probe showing deeper LSTMs overfit the sparse stage sequences).
- **Section 3.11 / Table 3.1 (p. 19-20):** The table is referenced but the full content is not visible in the provided text. Ensure it reports not just macro-F1 but also per-stage recall and inference latency for all three architectures.
- **Section 3.12-3.15 (pp. 20-23):** The MDP and reward function are the core technical contribution. The piecewise reward (Eq. 3.1 and 3.2) is complex. Add a **pseudocode block** (e.g., Algorithm 1) showing exactly how the reward is computed at each step. This will prevent ambiguity.
- **Section 3.16 (p. 24):** Justification of `p_de-esc = 0.6`. The sweep is good, but explain why the curve plateaus after 0.6. Is it because the attacker is too easily reset, making the MDP trivial?
- **Section 3.19 (p. 24-25):** Training protocol. Why 250k steps? The justification (probe-driven scaling) is fine, but show a small learning-curve figure in the methodology to justify this choice visually.
- **Section 3.20 (p. 25-26):** Baselines. The oracle is well-defined. However, the “Always-block” baseline achieves a **mitigated-impact rate of 1.000** (Table 4.5). This suggests that a trivial baseline actually solves the security task perfectly (by always blocking), albeit at low reward. This undermines the need for RL unless reward is the only metric. Discuss why always-block is unacceptable operationally (it blocks 100% of benign traffic). This is obvious, but must be stated explicitly to preempt reviewer objections.

### 4.6 Results and Discussion (Chapter 4)
- **Section 4.2 (p. 28-29):** Dataset audit. Move the leakage bug details to an appendix. Figure 4.1 is fine.
- **Section 4.3 (p. 30):** Red-team LSTM. Figure 4.2, panel (a): The caption says “cross-entropy loss and token accuracy” but the right y-axis is labeled **Macro-F1**, not token accuracy. **Fix the caption or the figure label.**
- **Section 4.4 (p. 32):** Stage detector. Figure 4.3 is clear. The OOD asymmetry (recall 0.001 vs. 0.999) is striking. Add a brief discussion of whether data augmentation or contrastive learning could address the VulnerabilityScan blind spot.
- **Section 4.5 (p. 34-36):** Blue-team training.
  - Figure 4.4: The x-axis goes to 500,000 steps, but the text says training stops at 250,000. **Clarify** whether the plot shows an extended run or if the x-axis label is wrong.
  - Figure 4.5: Action-distribution evolution. Good. Add a small table quantifying the final per-stage action proportions for PPO.
- **Section 4.6 (p. 37-41):** Held-out benchmark.
  - Figure 4.6: The red dashed line is labeled “Recommended-Action floor (1624)”. This is confusing: 1624 is the **oracle ceiling**, not a floor. **Relabel to “Oracle ceiling (1624)”.**
  - Figure 4.7: The colorbar is only shown on the A2C panel. Add a unified colorbar to all panels or ensure consistent color mapping. The red boxes (proportionality band) are excellent.
  - Figure 4.8: Latency CDFs. The x-axis is log-scale; good. However, the legend lists “Recommended-Action” and “RF-Acting” with similar line styles. Make RF-Acting visually distinct (e.g., thick dashed line) since it is the main deployable competitor.
  - Figure 4.9: This is an **image of a table**, not a native LaTeX table. **Replace with a proper typeset table.** Images of tables are unacceptable in journal submissions.
  - Table 4.4 (p. 40): Only three pairwise comparisons are shown. **Complete the table** with all 15 pairwise comparisons (or at least all DRL vs. baseline pairs). Welch’s t-test is appropriate, but add a Bonferroni correction note since multiple comparisons are being made.
  - Table 4.5 (p. 57): Add a column for **benign FPR** to make the deployability trade-off explicit.
- **Section 4.7 (p. 42-44):** Ablations.
  - Figure 4.10 (F9): The `impact_is_terminal=False` cell is the clear winner. This should be the **primary result**, not an ablation.
  - Figure 4.11 (F10): Good. Add error bars for the oracle rule if multiple seeds were run.
- **Section 4.8 (p. 45-46):** OOD robustness.
  - Figure 4.12: The caption contains internal audit tags (“audit-AF1”). Remove these.
  - The VulnerabilityScan result is the most important. Enlarge this panel or move it to the main text as a standalone figure with deeper analysis.
- **Section 4.9 (p. 47):** Threats to validity. This is a strong section, but it should be expanded to include:
  - **Simulator fidelity threat:** The red-team LSTM is trained on in-distribution transitions. Does it produce realistic *inter-arrival times* and *traffic volumes*, or only stage sequences? The defender observes per-flow features; if the feature realizer samples unrealistic vectors during stage transitions, the RL policy may overfit to synthetic feature-stage correlations.
  - **Reward function threat:** The entire policy is a function of six manually tuned coefficients. A sensitivity analysis (Section 4.7) is good, but the threat that *different security operators would tune these differently* is not addressed.

### 4.7 Conclusions (Chapter 5)
- **Section 5.1 (p. 48-49):** Contribution summary. As noted, demote the leakage bug and elevate the reward-function structural analysis.
- **Section 5.2 (p. 50):** “Findings Worth Defending.” This is an unusual section for a thesis and would be removed in a journal. The three findings are fine, but frame them as “Key Results” rather than a defensive crouch.
- **Section 5.3 (p. 51):** Limitations. Add: **“The system has not been tested on live traffic or in a physical testbed; all results are simulation-based on CICIoT2023.”**
- **Section 5.5-5.6 (p. 52-53):** Future work. The six directions are sensible. Prioritize Direction 6 (reward redesign with explicit FPR) because the current FPR makes Directions 2 and 4 (edge deployment, federated learning) premature.

### 4.8 Appendices
- **Appendix A (p. 57-60):** The reproducibility protocol is excellent. However, the smoke harness verifies *hashes*, not *semantic correctness*. Add one sentence acknowledging that hash verification ensures bit-exact reproduction but does not guarantee that the underlying code is bug-free.
- **Appendix B (p. 61):** Kill-chain mapping table. Ensure the `VulnerabilityScan` and `XSS` classes are clearly marked with the ⋆ symbol (they are, but verify in the final PDF).
- **Appendix C (p. 62):** Hyperparameters. Table C.1 is clear. Add the optimizer name (Adam) and whether weight decay was used.
- **Appendix D (p. 63-65):** Sensitivity sweeps.
  - Figure D.1 duplicates Figure 4.11. If this is intentional for appendix completeness, state so in the caption.
  - Figure D.2 caption says “DQN, seeds 0–2” but the figure title says “PPO, 3 seeds”. **Fix the inconsistency.**
  - Figure D.3 caption says “DQN, seed 0” but the figure title says “PPO, 3 seeds”. **Fix the inconsistency.**

---

## 5. Formatting, Language, and Style

### 5.1 Bilingual Artifacts
The thesis mixes Portuguese and English: “Resumo,” “Agradecimentos,” “Capítulo,” “Figura,” “Tabela.” This is standard for UNICAMP, but for an international journal submission, the entire manuscript must be in English. Headers like “Capítulo 1. Introduction” should be “Chapter 1. Introduction.”

### 5.2 Mathematical Notation
- Equation (3.1) and (3.2) are rendered as inline text with awkward spacing (e.g., `R_{t}\;=\;(\mathrm{s u m~o f~t h e~s i x~c o m p o n e n t s~a b o v e})`). Ensure LaTeX math mode is clean.
- The MDP tuple on p. 23 has extraneous symbols (`(S,bar A P\bar{,}R,\gamma)`). Clean up all math macros.

### 5.3 Figures and Tables
- **All tables must be native LaTeX (or Word) tables, not images.** Figure 4.9 is the worst offender.
- **Figure captions:** Ensure every caption is self-contained. E.g., Figure 4.4 should state “Training runs to 250,000 steps; curves shown to 500,000 steps include a post-hoc extension to verify plateau” (if that is true).
- **Color palette:** Use colorblind-safe palettes (e.g., viridis, Okabe-Ito). The red/green/blue in Figure 4.3 may be problematic for deuteranopia.

### 5.4 Citations
- The citation style is inconsistent (some use “et al.” in italics, some do not). Standardize to the target venue’s format (IEEE uses [1], [2] numbering).
- Several claims cite industry reports (Morgan 2025, Vailshery 2025). These are acceptable for market size claims but should be supplemented with academic citations where possible.

---

## 6. Actionable Recommendations: Re-run, Re-evaluate, Remove, Steer

### 6.1 Must Re-run / Re-evaluate
1. **Primary benchmark under `impact_is_terminal=False`:** Re-run the full held-out benchmark (Table 4.3, Figure 4.6, 4.7, 4.8, 4.9) with the fixed contract. Report these as the main results.
2. **PPO trained with `stage_pred` input:** Add an observation ablation where the agent sees the MLP stage detector’s output during training. Run 5 seeds, 250k steps. Compare to the oracle to see if the gap is due to state uncertainty.
3. **RF-Acting with {10, 20, 50} trees:** Re-run inference latency and reward for smaller forests to fairly characterize the latency–reward frontier.
4. **Complete pairwise statistical tests:** Fill Table 4.4 with all policy pairs. Apply Bonferroni correction.
5. **Benign-FPR constrained policy:** Re-run PPO with an added penalty term `-beta * FPR` (tune `beta` to target <1% FPR) and report the reward degradation. If this is too much for the thesis timeline, simulate it via post-hoc threshold tuning on the trained policy’s action probabilities.

### 6.2 Must Remove or Condense
1. **Leakage bug narrative in main text:** Reduce to one sentence; move details to Appendix A.
2. **Gate scoreboard jargon (G5.4, G6.2, D7.9.1, etc.):** Remove from Chapters 4 and 5. Keep only in Appendix A.
3. **“Findings Worth Defending” section (5.2):** Remove or reframe as “Principal Results.”
4. **Figure 4.9 (image-of-table):** Replace with native LaTeX table.
5. **Duplicate appendix figures (D.1 vs. 4.11):** Either remove the duplicate or explicitly label it “Appendix reproduction of Figure 4.11 for reference.”

### 6.3 Must Steer / Reframe
1. **Narrative arc:** Shift from *“We built a deployable defense system”* to *“We rigorously benchmarked DRL for kill-chain defense and discovered that (a) reward mis-specification is the dominant failure mode, (b) latency advantages are real but benign FPR is the blocking issue, and (c) OOD generalization remains unsolved.”* This is more defensible and academically interesting.
2. **Contribution list:** Replace Contribution #2 (leakage bug) with something like: *“A structural analysis of reward mis-specification in security MDPs, showing that terminal-impact semantics dominate policy behavior more than coefficient tuning.”*
3. **OOD discussion:** Steer from “robust to, not better at” to “static baselines can outperform RL on feature-novel attacks, revealing a brittleness that train-time augmentation must address.”
4. **Deployability claims:** Add the qualifier **“potentially deployable pending FPR reduction”** whenever latency or model size is praised.

---

## 7. Final Verdict

**Recommendation: Major Revision Required**

The thesis demonstrates strong engineering, an excellent reproducibility protocol, and intellectual honesty in reporting the reward mis-specification. However, it currently **sells the wrong result**: the default-contract policy (29.7% impact mitigation, 10% benign FPR) is presented as a success story. Until the manuscript is restructured around the fixed-contract results, the benign-FPR problem is elevated from a footnote to a primary limitation, and the missing ablations (detector-input RL, smaller RF forests) are added, this work risks rejection at any top-tier venue for over-claiming and under-verifying deployability.

**If the author executes the major revisions above—particularly the contract pivot and the FPR analysis—this work will be a solid candidate for *IEEE Transactions on Dependable and Secure Computing* (TDSC) or *ACM CCS*.**

---

*Reviewer Signature:*
*Specialist in RL, IoT Security, and Empirical ML*
*Date: 2026-05-25*
