# Attack Sequence Generator (LSTM)

## Purpose

The Attack Sequence Generator is the **Red Team** model. It learns a grammar over Kill Chain stages and generates plausible attack progressions for the RL environment.

## Model architecture

From `src/generator/attack_sequence_generator.py`:

- Input: stage IDs $\in \{0,1,2,3,4\}$
- Embedding: $\mathbb{R}^{\text{num\_stages}} \to \mathbb{R}^{d}$
- Stacked LSTM: hidden size $h$, layers $L$
- Output: Dense layer to logits over 5 stages

$$\text{logits}_t = W h_t + b$$

### Default configuration

- `embedding_dim = 32`
- `hidden_size = 64`
- `num_layers = 2`
- `dropout = 0.1`
- `temperature = 1.0`

## Training data

Episodes are generated via `EpisodeGenerator` (synthetic but grammar-constrained). The model is trained as a **next-token predictor**:

Given a window $x_{t-k:t-1}$, predict $x_t$.

$$\mathcal{L} = - \log p\left(x_t \mid x_{t-k:t-1}\right)$$

## Episode generation (grammar)

Kill Chain rules implemented in `src/generator/episode_generator.py`:

- **Progression**: $P(S_{t+1} > S_t) > 0$
- **Persistence**: $P(S_{t+1} = S_t) > 0$
- **No regression**: $S_{t+1} < S_t$ not allowed (except external reset)

Episodes can optionally incorporate dataset stage distributions with **Laplace smoothing** and **temperature flattening**.

### Temperature on stage distribution

Let $p_i$ be the smoothed probability for stage $i$.

$$\tilde{p}_i = \frac{p_i^{\tau}}{\sum_j p_j^{\tau}}$$

- $\tau < 1$ flattens imbalance.
- $\tau > 1$ sharpens imbalance.

### Minimum coverage

`min_stage_coverage` forces a minimum fraction of episodes containing rare stages (e.g., ACCESS) by regenerating samples until coverage is met.

## Transition masks

`TransitionMask` (optional) enforces grammar at inference:

- Blocks regression (except IMPACT → BENIGN reset).
- Always allows persistence and forward progression.

Used by `AttackSequenceGenerator.set_transition_mask(mask)`.

## Outputs

Training produces:

- `attack_sequence_generator.pth`
- `config.json` (model config)
- `training_config.json`
- `loss_curves.png`

## Key files

- `src/generator/attack_sequence_generator.py`
- `src/generator/episode_generator.py`
- `src/generator/transition_mask.py`
- `src/training/generator_trainer.py`
