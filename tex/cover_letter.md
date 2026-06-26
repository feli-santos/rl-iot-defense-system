# Cover Letter

**To:** The Editor-in-Chief
**Journal:** *Future Internet* (MDPI) — Section: *Cybersecurity*
**Re:** Submission of original research manuscript

---

Dear Editor,

We are pleased to submit our manuscript, **"A Kill-Chain-Aware Reinforcement
Learning Defense Framework for IoT Networks,"** for consideration as an original
research article in *Future Internet*.

## What problem we address

Reinforcement learning (RL) is increasingly proposed for autonomous IoT defense,
but a recurring and under-examined objection is that an RL "defender" is often
just a supervised intrusion classifier in disguise: if every decision can be made
from the current flow alone, a per-flow classifier suffices and the RL machinery
adds nothing. Our manuscript confronts this objection directly rather than
sidestepping it.

## Core contribution

We cast kill-chain IoT defense as a **genuine partially observable Markov decision
process (POMDP)**. Using **real CICIoT2023 feature rows** (105 physical IoT
devices) projected onto a five-stage Cyber Kill Chain, we build a reactive
*tug-of-war* attacker whose escalation pressure is coupled to its proximity to the
impact stage, and we make the defender act **without ever observing the true
stage**. A single tunable parameter, the observation aliasing rate $\alpha$,
controls how ambiguous each individual observation is, so the stage can be
inferred only across a temporal window.

This design yields a falsifiable claim and three supporting results:

1. **Aliasing crossover.** At $\alpha=0$ (fully observable) the windowed PPO agent
   and a tuned detector-coupled Random-Forest policy **tie** — proving the
   environment does not favor RL by construction. As $\alpha$ rises, the memoryless
   classifier degrades monotonically while PPO holds flat, opening a gap with
   disjoint 95% confidence intervals from $\alpha=0.4$ onward.

2. **Reward-coupling ablation.** Under both a privileged-information (coupled) and
   a sparse outcome-only reward, the best RL agent beats the supervised baseline —
   neutralizing the objection that a privileged reward reduces the task to
   classification.

3. **Zero-day robustness.** On ten held-out attack classes, PPO prevents a
   uniformly larger fraction of attacks than the supervised baseline across **all
   ten**, and this advantage does **not** track detector recall — it is a
   temporal-control property a memoryless classifier cannot capture.

## Why *Future Internet*

The manuscript sits squarely within the journal's Cybersecurity scope: autonomous
network defense, IoT security, and applied machine learning. CICIoT2023 was itself
introduced in an MDPI venue, and our work advances the responsible-evaluation
conversation around RL-for-security by foregrounding when RL *does and does not*
help.

## Rigor and reproducibility

Every quantitative claim carries bootstrap confidence intervals over ten seeds
(300 episodes per policy). All figures are regenerated from canonical result JSONs
through a hash-chained manifest (Git commit + SHA-256 of inputs and outputs), and
the full pipeline is verified by an automated test suite. We disclose limitations
candidly, including that the RL advantage is conditional on partial observability,
that the supervised baseline is memoryless by construction, and that evaluation is
presently single-dataset (with Bot-IoT named as the natural replication target).

## Declarations

This manuscript is original, has not been published elsewhere, and is not under
consideration by any other journal. The authors declare no competing interests.

We believe this work will be of interest to the readership of *Future Internet*
and look forward to your editorial assessment.

Sincerely,

Felipe Augusto Oliveira dos Santos and Prof. Dr. Denis Fantinato
School of Electrical and Computer Engineering (FEEC), University of Campinas (UNICAMP)
