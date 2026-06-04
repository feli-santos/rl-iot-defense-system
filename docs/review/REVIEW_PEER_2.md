# IEEE/Journal Level A Peer Review Report

**Manuscript/Thesis Title:** An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning
**Author:** Felipe Augusto Oliveira dos Santos
**Advisor:** Prof. Dr. Denis Fantinato
**Institution:** Faculty of Electrical and Computer Engineering (FEEC), State University of Campinas (UNICAMP)
**Degree:** Master in Electrical Engineering (Computer Engineering Area)
**Reviewer Category:** Level A Expert Review (Reinforcement Learning, IoT Security, and Network Cyber-Defense)

---

## 1. Overall Executive Summary
The thesis addresses a highly relevant, timely, and complex problem: the creation of autonomous, adaptive, and low-latency security mechanisms for resource-constrained Internet of Things (IoT) devices. By combining proactive attack-stage prediction with Deep Reinforcement Learning (DRL), the author designs a closed-loop defense framework capable of operating at the network layer. The framework maps network flow features onto a 5-stage Cyber Kill Chain and treats defense as a sequential decision-making problem.

The submission is exceptionally rigorous in its engineering practices, particularly concerning reproducibility, auditability, and honesty regarding data preparation bugs (e.g., the discovered data leakage issue). However, a critical peer-review inspection reveals several **major data/figure contradictions**, mathematical and conceptual limitations in the reward structures, and severe operational deployment constraints (e.g., prohibitive false positive rates) that must be comprehensively addressed before journal publication or final academic approval.

---

## 2. Key Strengths of the Work
1. **Engineering and Reproducibility Rigor:** The integration of a cryptographic reproducibility protocol utilizing SHA-256 hash chains pinned to Git commits and verified via an automated smoke-test harness (`reproducibility_smoke`) is a gold standard rarely seen in machine learning theses.
2. **Methodological Honesty:** The transparent reporting and post-mortem analysis of the out-of-distribution (OOD) data leakage bug (commit `3cd2fb9`) reflects high scientific integrity and adds educational value to the methodology section.
3. **Identification of Reward Mis-specification:** The diagnosis of "de-escalation bonus farming" is an excellent, advanced RL finding. Recognizing that the agent intentionally accepted terminal exploit losses to optimize immediate proxy rewards demonstrates a sophisticated understanding of credit assignment and reward hacking.
4. **Latency Frontier Analysis:** The structural analysis of the latency-reward trade-off establishes a compelling real-world use case for DRL models (~4K parameters, sub-millisecond inference) over bulky supervised ensembles (100-tree Random Forest) that cause high packet-queuing delays.

---

## 3. Critical Major Weaknesses and Discrepancies (Required Corrections)

### A. Severe Numerical and Labeling Contradictions Between Figures and Tables
There are profound, irreconcilable discrepancies between the numbers plotted in the summary figures and those reported in the text and data tables. This strongly implies that the author updated the experimental scripts or dataset splits but failed to regenerate all figures uniformly, or vice-versa.

* **Figure 4.6 vs. Table 4.3 and Figure 4.9 (Table F5) Discrepancies:**
    * In **Figure 4.6** (page 54), the mean episodic reward for **DQN** is visibly plotted and textually labeled as `1336 (1265, 1408)`, and **PPO** is labeled as `1313 (1253, 1372)`. This would mean DQN outperforms PPO.
    * In **Table 4.3** (page 56) and **Figure 4.9** (page 56), the hierarchy is completely inverted: **PPO** is listed as the best deployable policy with a mean reward of `+1334.5 [1317.3, 1352.3]`, whereas **DQN** is listed as significantly lower at `+1218.9 [1159.4, 1272.4]`.
    * For **RF-Acting**, Figure 4.6 lists the reward as `1508 (1455, 1565)`, whereas Table 4.3 lists it as `+1486.2 [1430.2, 1538.4]` and Figure 4.9 lists it as `1486`.
    * *Reviewer Note:* The data labels for DQN and PPO appear to have been swapped or misaligned in the plotting script for Figure 4.6. This undermines the technical validity of the primary results section.
* **Training Timestep Inflation in Figures:**
    * The text in **Section 3.19** explicitly states: *"Each algorithm is trained for 250,000 environment steps... the choice of 250,000 rather than 500,000 was made on the basis of a probe-driven scaling study... marginal return became negligible past the 250,000-step horizon."*
    * However, the x-axes of **Figure 4.4** (Blue-team learning curves) and **Figure 4.5** (Action-distribution evolution) both clearly show timelines extending up to **500,000 timesteps** (`500000`).
    * *Reviewer Note:* If training was cut off at 250k steps, the figures should not show active data up to 500k steps. This suggests either a major text-figure mismatch or that the scaling study was not adhered to in the final run.

### B. Theoretical Paradox of the Supervised Baseline (RF-Acting)
* The primary benchmark indicates that **RF-Acting** (the fully deployable supervised Random Forest classifier coupled with the static recommended-action mapping) yields an absolute mean episodic reward of **+1486.2**, which is **substantially higher** than the best primary DRL agent (PPO at +1334.5).
* From a pure reinforcement learning perspective, this indicates that the model-free DRL agents failed to discover a policy superior to a basic greedy classification framework mapped to an intuitive heuristic rule.
* While the author frames this via a "latency-reward trade-off frontier" (which is a valid architectural defense), the scientific claim that DRL provides a "robust mathematical foundation to outmaneuver attackers" is weakened if it cannot beat a static rule-table fed by an off-the-shelf classifier in an in-distribution setting.

### C. Prohibitive False Positive Rates (FPR) for Operational Deployment
* The trained agents exhibit a benign-traffic false positive rate (FPR) ranging from **9.4% to 12.7%** (applying BLOCK or ISOLATE on benign traffic).
* In real-world enterprise or industrial network security, an FPR exceeding **1%** is considered production-breaking, as it causes massive operational disruption and self-inflicted Denial-of-Service.
* The author treats this as a side-note limitation resulting from reward farming. However, from a security reviewer standpoint, a system that drops ~10% of legitimate business traffic to preserve an internal reward metric is fundamentally broken. This severe limitation needs a much harsher and realistic critique in the discussion section.

### D. Oversimplification via the Monotonic Attacker Assumption
* The threat model assumes the attacker can only advance forward through the kill chain (`BENIGN -> RECON -> ACCESS -> MANEUVER -> IMPACT`) and never retreat, loop, or pivot.
* While this bounds the scope, it transforms the underlying MDP into a highly simplified sequential alignment task. Real-world advanced persistent threats (APTs) rely heavily on non-monotonic movements, multi-wave campaigns, and stealthy low-and-slow execution. This assumption significantly limits the claim of the framework being "highly adaptive to complex cyber-attacks."

---

## 4. Comprehensive Section-by-Section Review

### Title, Front Matter, and Formatting
* **Title:** Clear and representative of the technical architecture.
* **Resumo & Abstract:** Good technical summaries, though they must be updated once the numerical contradictions between 250k vs. 500k timesteps and the PPO/DQN values are resolved.
* **Placeholders:** Pages 3 and 4 contain explicit placeholders (`INCLUA AQUI O PDF COM A FICHA CATALOGRÁFICA` and `FOLHA DE ASSINATURAS`). While acceptable for a draft review, the final text must note these as pending administrative additions.

### Chapter 1: Introduction
* **Contextualization:** Excellent utilization of contemporary security citations (e.g., Morgan 2025; Vailshery 2025) and a clear problem statement emphasizing the resource constraints of IoT edge nodes.
* **Contributions:** Clearly articulated, but Contribution #3 (*Comparative DRL benchmark with a 141x latency advantage*) contains the disputed PPO inference latency and reward values that conflict with Figure 4.6.

### Chapter 2: Background and Related Work
* **Foundational Soundness:** The formulation of the Markov Decision Process (MDP) and the recursive Bellman optimality equations is technically flawless and cleanly rendered using standard mathematical notation.
* **Literature Positioning:** Table 2.1 provides an exceptional, granular head-to-head architectural comparison against *IoTWarden*, *HoneyIoT*, and *Nguyen et al.* This positions the thesis well within the current state of the art by emphasizing real dataset captures over synthetic models.

### Chapter 3: Methodology
* **Feature Realization:** The 29-dimensional flow mapping from `CICIoT2023` to the 5-stage Cyber Kill Chain is logical and supported by Appendix B.
* **Gymnasium Environment Lifecycle:** The implementation details of `AdversarialIoTEnv` are well explained. However, the *Lifecycle-floor impact clamp* (clamping IMPACT to MANEUVER before step 20) introduced in Section 3.14 is highly artificial. It forces an unnatural baseline bias into the Mean Time To Compromise (MTTC) metric, which the author acknowledges but does not fully justify mathematically.
* **Reward Parameters:** Table 3.2 lists the config weights. The penalties are massive (e.g., Overreact = -50, Block-on-Benign = -100). These heavy hand-tuned constraints explain why the RL agent struggled to explore freely and instead fell into degenerate false-positive behaviors or reward farming.

### Chapter 4: Results and Discussion
* **The Structural Core of Revisions:** This chapter requires heavy editing due to the data alignment bugs highlighted in Section 3.
* **Figure 4.2 (LSTM Red Team):** The transition matrices match the priors well, validating the synthetic attacker.
* **Figure 4.3 (Stage Detector):** Captures the OOD asymmetry cleanly. The structural blindness on `VulnerabilityScan` (0.001 recall) is a valuable machine learning finding.
* **Section 4.5.0.0.2 (Reward Mis-specification):** This is the strongest intellectual contribution of the text. The explanation of how the PPO agent maximizes the proxy objective (+1575 from de-escalation bonuses) while ignoring the intended human objective is a phenomenal case study in AI alignment.
* **Ablation (impact_is_terminal = False):** The structural modification that allows the agent to act *during* the IMPACT stage (lifting the mitigated impact rate from 0.153 to 0.900) is highly compelling and should be elevated as the primary architectural recommendation of the thesis, rather than just an ablation variant.

### Chapter 5: Conclusions and Future Work
* **Summary:** Accurately reflects the findings, but repeats the text/figure contradictions.
* **Future Directions:** The recommended paths are excellent, particularly **Direction 6** (*Reward-function redesign with explicit FPR constraints*) and **Direction 2** (*Edge-hardware deployment*). These directly address the architectural vulnerabilities identified during the review.

---

## 5. Actionable Blueprint for Adjustments (Instructions for the Next Agent)

To bring this thesis to a publishable, defensible Level A status, the executing agent must implement the following specific modifications:

### 1. Re-run or Re-generate Data Visualizations (Highest Priority)
* **Triage Figure 4.6:** Check the underlying plotting script (likely in `scripts/visualize_benchmark.py` or `docs/results/benchmark/`). Determine why Figure 4.6 lists DQN at 1336 and PPO at 1313 while Table 4.3 and Figure 4.9 state PPO is 1335 and DQN is 1219. Correct the data array bindings or labels so they match the tables exactly.
* **Fix the Timestep Axis Mismatch:** If the models were truly trained for 250k steps (as justified by the scaling study), crop the x-axis of Figure 4.4 and Figure 4.5 to `250,000`. If they were trained for 500k steps, rewrite Section 3.19 and Chapter 4 text to accurately reflect the 500k step horizon.

### 2. Deepen Technical Critiques and Discussion Prose
* **Expand on the RF-Acting Paradox:** Add a dedicated paragraph in Section 4.9 explaining *why* model-free DRL agents could not surpass the reward of the supervised RF-Acting policy. Frame it as a limitation of model-free exploration in highly constrained, step-dependent environments, emphasizing that the value of DRL lies in its sub-millisecond execution frontier rather than reward dominance.
* **Strengthen the False Positive Critique:** Revise the "Limitations" section (Section 5.3, Point 4). Explicitly state that an FPR of 9.4%-12.7% renders the current primary policy configuration *completely undeployable* in production environments. Explicitly connect this to the necessity of the proposed Future Work Direction 6 (Lagrangian constrained MDPs).

### 3. Editorial and Textual Polishing
* Change the typo `OQN` in Figure 4.10 text extraction (Page 59, row `OQN+1336`) to `DQN`.
* Ensure that any reference to the `impact_is_terminal = False` ablation is framed as the *optimized structural architecture*, emphasizing that future work should default to this setting to eliminate reward hacking from the outset.

---
**Review Verdict:** *Major Revision Required.* The engineering pipeline and structural findings are excellent, but the data contradictions within the results visual matrix must be rectified immediately.
