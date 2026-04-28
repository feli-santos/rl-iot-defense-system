# Figure F1 — LSTM Red Team learning curves

**LaTeX caption (proposed).**
*Training and validation cross-entropy loss (left) and validation macro-F1
on a 200-sample-per-stage **balanced** subset (right) for the LSTM Red
Team episode generator. The model is a 1-layer LSTM with hidden size 32,
embedding 16, dropout 0.2, trained for 15 epochs on 50 000 synthetic
attack episodes. Best balanced-validation loss = 0.854 at epoch 1.*

**What to look for.**
1. Train loss converges fast — by epoch 5 the model has captured the 5-token
   transition grammar. The flat tail beyond epoch 5 indicates *successful
   compression* of the synthetic episodes, not memorization (see G1 below).
2. Balanced-validation loss tracks train loss with a constant ≈ 0.87 offset.
   The offset is the **distribution-mismatch penalty**: balanced val
   over-samples rare stages 1–3, while train mirrors the natural
   distribution. The model is *not* overfitting — see exit gate **G1**:
   the i.i.d. holdout gap is 0.035 (3.5 %), well under the 25 % threshold.
3. Macro-F1 plateaus around 0.44 because the balanced eval is unforgiving on
   the rare stages where the LSTM has very few exemplars; on natural-
   distribution holdout, macro-F1 = 0.487 and token accuracy = 97.65 %.

**Phase-2 exit gates (PLAN.md §3.2).**
| Gate | Threshold | Observed | Result |
|------|-----------|---------:|--------|
| G1 i.i.d. train↔holdout loss gap | ≤ 0.25 | 0.035 | ✅ |
| G2 token accuracy on holdout    | ≥ 0.55 | 0.977 | ✅ |
| G3 KL(P_lstm ‖ P_truth) on the 5×5 transition matrix | ≤ 0.05 | 0.021 | ✅ |
| G4 cosine(stage-freq LSTM, truth rollouts) | ≥ 0.90 | 1.000 | ✅ |

**How it was generated.**
`PYTHONPATH=. python -m scripts.red_team.train_lstm --no-mlflow` (seed 42,
git SHA recorded in `manifest.json`).
