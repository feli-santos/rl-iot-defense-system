# Design Decisions

## 1) Real data only

The environment samples **actual CICIoT2023 rows** via `RealizationEngine`. No synthetic traffic is created for the observation space, preserving realism.

## 2) Kill Chain abstraction

All labels are mapped to a **5-stage Kill Chain** for tractable sequence modeling and defense escalation.

## 3) LSTM generator over grammar rules

The Red Team uses an LSTM next-token model trained on grammar-based episodes to simulate realistic escalation patterns while remaining interpretable and tunable.

## 4) Partial observability

The agent does not see the true stage. Observations are feature windows; stage is included only in `info` for evaluation.

## 5) Reward shaping to balance security & availability

Large IMPACT penalty + action costs + benign penalties drive conservative but timely defense.

## 6) Stable Baselines3 as RL framework

Standard SB3 algorithms and MlpPolicy are used for reproducibility and common baselines.

## 7) MLflow tracking

Training runs produce rich telemetry for auditability and post-hoc analysis.

## 8) Explicit benchmarking layer

A dedicated benchmarking module computes **security metrics**, not just reward, to better align with operational goals.
