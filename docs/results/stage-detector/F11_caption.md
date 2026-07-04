**F11.** Tuned RandomForest stage detection on the balanced test split
(1 000 rows / stage). Left: per-class F1 for the tuned RandomForest
stage detector (the model underlying the RF-Acting baseline); the dashed
line marks its macro-F1. Right: row-normalised confusion matrix of the
same detector on the same split. The diagonal-heavy structure shows the
detector correctly identifies most stages, with the bulk of confusion
concentrated at RECON↔ACCESS — the ambiguous middle of the kill chain
the RL agent must act through. Per-stage and per-attack-class numbers
are committed in `F11_summary.json`.
