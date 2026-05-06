# F3 — RL Blue Team learning curves

Three-panel learning curve over training timesteps for DQN (red),
PPO (blue), and A2C (green), each averaged across **5 seeds** with
shaded **95 % bootstrap CI** bands. Solid lines are training-time
metrics from `episodes.jsonl`; dotted lines are deterministic
evaluation rollouts on the held-out `val_balanced` split (eval every
25 K timesteps × 30 episodes per block, written to `eval.jsonl`).

**Panels (left → right):**

1. **Mean episodic reward.** All three algorithms converge to a
   strongly positive reward of ~+1300 per episode by ~150 K
   timesteps. The oracle recommended-action policy (Phase-3 G3.4)
   nets ~+50 per episode on `val_balanced`, so the trained agent
   beats the hand-crafted policy by **roughly 25×** — confirming the
   env exposes a learnable structure (G5.2 PASS).
2. **Mean MTTC (Mean Time-To-Compromise).** Stable at **19.2-19.3
   steps** across all algorithms — the IMPACT-clamp floor
   (`min_episode_length=20`) holds in practice (G5.3 PASS at the
   threshold MTTC ≥ 19 from D5.4.1).
3. **Mitigated-impact rate.** Fraction of episodes ending with
   `end_outcome == "impact_mitigated"` (BLOCK or ISOLATE on the
   IMPACT step). All three algos land at **~0.25**, well below the
   D5.4.1-revised G5.4 target of ≥ 0.5 — see RESULTS.md §4 Finding 2
   for the thesis-relevant interpretation.

Per-algo last-10 % numbers and per-seed values are in
`F3_summary.json`; SHA-256 inputs/outputs in `F3_manifest.json`.
