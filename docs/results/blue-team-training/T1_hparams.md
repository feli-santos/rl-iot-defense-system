# T1 — Blue-Team Per-Algorithm Hyperparameters

Generated from 30 runs across 3 algorithms (seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).

All values are PLAN §8 D5.4 defaults. Ablation & Robustness may revisit
hyperparameters; Blue-Team Training reports them as a frozen reference.

| algo | total_timesteps | learning_rate | n_steps | gamma | gae_lambda | ent_coef | vf_coef | max_grad_norm | buffer_size | learning_starts | batch_size | tau | target_update_interval | exploration_fraction | exploration_initial_eps | exploration_final_eps | n_epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a2c | 1500000 | 7e-04 | 256 | 0.990 | 0.950 | 0.010 | 0.500 | 0.500 |  |  |  |  |  |  |  |  |  |
| dqn | 1500000 | 5e-04 |  | 0.990 |  |  |  |  | 200000 | 5000 | 64 | 1 | 5000 | 0.200 | 1 | 0.050 |  |
| ppo | 1500000 | 3e-04 | 2048 | 0.990 | 0.950 | 0.010 | 0.500 | 0.500 |  |  | 64 |  |  |  |  |  | 10 |
