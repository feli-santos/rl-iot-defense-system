**F11.** Per-stage detection recall on the balanced test split (1 000
rows / stage). Left: stage-recall comparison across the production MLP
detector (blue) and RandomForest baseline (orange); the dashed line
marks the best macro-F1 achieved by either model.
Right: row-normalised confusion matrix of the production detector on
the same split. The diagonal-heavy structure shows the detector
correctly identifies most stage transitions, with the bulk of confusion
concentrated near MANEUVER↔IMPACT — exactly the boundary the RL agent
will have to act on. Per-stage and per-attack-class numbers, plus
results on the full (BENIGN-heavy) test split, are committed in
`F11_summary.json`.
