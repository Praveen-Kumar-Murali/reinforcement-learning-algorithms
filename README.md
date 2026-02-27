# Reinforcement Learning Algorithms

> A comprehensive implementation of foundational to advanced Reinforcement Learning algorithms in Python, including DP, Q-Learning, DQN, REINFORCE, and Imitation Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg?style=flat)
![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-green.svg?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)

## Algorithms

| Category | Algorithm | Module |
|---|---|---|
| Dynamic Programming | Iterative Policy Evaluation, Policy Iteration, Value Iteration | `src/agents/dp_agent.py` |
| Model-Free RL | Monte Carlo, Q-Learning, SARSA | `src/agents/model_free_agent.py` |
| Deep RL (Value-Based) | DQN, Double DQN, Dueling DQN | `src/agents/dqn_agent.py` |
| Deep RL (Policy-Based) | REINFORCE (with optional baseline) | `src/agents/pg_agent.py` |
| Imitation Learning | Behavior Cloning | `src/agents/bc_agent.py` |

## Project Structure

```
reinforcement-learning-labs/
├── data/
│   └── datasets/               # Expert demonstration datasets (BC)
├── notebooks/                  # Interactive Jupyter notebooks
├── src/
│   ├── envs/
│   │   └── map_env.py          # Custom grid-world environment
│   ├── agents/
│   │   ├── dp_agent.py         # Dynamic Programming
│   │   ├── model_free_agent.py # MC, Q-learning, SARSA
│   │   ├── dqn_agent.py        # DQN / DDQN / Dueling DQN
│   │   ├── pg_agent.py         # REINFORCE (VPG)
│   │   └── bc_agent.py         # Behavior Cloning
│   ├── utils/
│   │   └── rl_aux.py           # Visualisation helpers
│   └── main.py                 # Unified CLI runner
├── report/
│   ├── project_report.md
│   └── result_gifs/
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Running Algorithms

All algorithms are accessible through the unified CLI runner:

```bash
# Dynamic Programming (Value Iteration on a 4×4 grid)
python src/main.py dp --algorithm value_iteration --size 4

# Model-Free Q-Learning on FrozenLake
python src/main.py model_free --algorithm q_learning --episodes 10000

# DQN on LunarLander-v2
python src/main.py dqn --episodes 500

# Double DQN with Dueling architecture
python src/main.py dqn --episodes 500 --double --dueling

# Policy Gradient (REINFORCE) with baseline
python src/main.py pg --episodes 2000 --baseline

# Behavior Cloning from expert demos
python src/main.py bc --dataset data/datasets/lunar_lander_10.npz --epochs 100
```

Run `python src/main.py <command> --help` for full option details.

## Environments

- **MapEnv** – Configurable grid-world with obstacles, stochastic transitions, and customisable size. Used by DP agents.
- **Gymnasium** – Standard benchmarks (`FrozenLake-v1`, `LunarLander-v2`) used by Model-Free, DQN, PG, and BC agents.

## Key Concepts Demonstrated

- **Dynamic Programming**: Exact solutions when the environment model is known.
- **Temporal-Difference Learning**: Online, model-free learning through bootstrapping.
- **Experience Replay & Target Networks**: Stabilising deep RL training.
- **Double DQN**: Reducing overestimation bias.
- **Dueling Architecture**: Separating state-value and advantage streams.
- **Policy Gradients**: Learning parameterised policies directly.
- **Variance Reduction**: Value-function baseline.
- **Imitation Learning**: Supervised learning from expert demonstrations.

## License

This project is released for educational purposes.
