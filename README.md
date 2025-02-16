# IoT Defense System powered by Reinforcement Learning

## Overview

This project implements a custom Gymnasium environment that simulates an IoT network defense scenario using Reinforcement Learning. The environment models a smart IoT network where an agent must take appropriate defensive actions (such as monitoring or blocking) to mitigate simulated attack conditions.

## Features

- **Custom Gymnasium Environment**: Simulates IoT network defense with dynamic state transitions.
- **Enhanced Reward Function**: Incorporates both the correctness of the action and the change in attack progression.
- **Mode Switching**: In defense mode, restricts the allowed actions to mimic realistic defense scenarios.
- **Reinforcement Learning**: Leverages Stable Baselines3 to train the agent on this environment.

## Requirements

- Python 3.12
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install 'stable-baselines3[extra]'
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---