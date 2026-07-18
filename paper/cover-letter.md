# Cover letter — *Internet of Things* (Elsevier)

Dear Editors,

We submit our manuscript, **"Partially Observable Kill-Chain Defense: Deep
Reinforcement Learning for Autonomous IoT Security,"** for consideration as a
full research article in *Internet of Things*.

The Internet of Things has produced a vast attack surface on devices too
resource-constrained for conventional security, motivating adaptive, autonomous
defense. Rather than asking merely *whether* reinforcement learning (RL) helps,
we ask the sharper question of *when* a learned sequential policy actually
outperforms a memoryless supervised classifier, and when any apparent advantage
is an artifact of task framing. We believe this fits squarely within the journal's
scope on Artificial Intelligence of Things, explainable machine learning for
IoT, and IoT security, reliability, and privacy.

Our contributions are:

1. A partially observable, kill-chain-aware IoT defense environment built on
   real CICIoT2023 traffic, with a reactive, bi-directional escalation attacker
   and a tunable observation-aliasing knob.
2. A controlled crossover experiment: at zero aliasing the windowed RL policy
   and the memoryless classifier **tie** (showing the environment is not rigged
   for RL), and RL overtakes the classifier with disjoint confidence intervals
   as aliasing grows.
3. A reward-coupling ablation showing the best RL agent outperforms the
   classifier under both sparse and shaped rewards, refuting the
   privileged-reward objection.
4. An out-of-distribution study over ten held-out attack classes, with no
   detectable dependence of the RL advantage on detector recall.
5. An honest algorithm-reliability account and a fully reproducible artifact
   chain (hash-chained manifests; all figures and numbers regenerable).

The manuscript is not under consideration elsewhere. All authors have approved
the submission and declare no competing interests.

We confirm the manuscript follows the journal's author guidelines, including a
CRediT statement, a declaration of generative-AI use, a competing-interests
declaration, a funding statement, and a data-availability statement (raw
CICIoT2023 is distributed under the CIC license by its original providers, so
we archive our code and hash-chained result manifests instead).

Thank you for your consideration.

Sincerely,

Felipe Santos (corresponding author)
School of Electrical and Computer Engineering (FEEC), University of Campinas
(UNICAMP), Campinas, São Paulo, Brazil
