# T1 — Phase-5 Per-Algorithm Hyperparameters

Generated from 15 runs across 3 algorithms (seeds: [0, 1, 2, 3, 4]).

All values are PLAN §8 D5.4 defaults. Phase 8 may revisit
hyperparameters; Phase 5 reports them as a frozen reference.

| algo | total_timesteps | learning_rate | n_steps | gamma | gae_lambda | ent_coef | vf_coef | max_grad_norm | buffer_size | learning_starts | batch_size | tau | target_update_interval | exploration_fraction | exploration_initial_eps | exploration_final_eps | n_epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a2c | 250000 | 7e-04 | 5 | 0.990 | 1 | 0 | 0.500 | 0.500 |  |  |  |  |  |  |  |  |  |
| dqn | 250000 | 1e-03 |  | 0.990 |  |  |  |  | 50000 | 1000 | 32 | 1 | 1000 | 0.100 | 1 | 0.050 |  |
| ppo | 250000 | 3e-04 | 2048 | 0.990 | 0.950 | 0.010 | 0.500 | 0.500 |  |  | 64 |  |  |  |  |  | 10 |
