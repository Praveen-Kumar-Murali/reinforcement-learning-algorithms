# Reinforcement Learning Labs – Technical Report

## 1. Introduction

This project implements a comprehensive suite of Reinforcement Learning (RL) algorithms, ranging from classical Dynamic Programming to deep neural network–based methods and imitation learning. Each algorithm is encapsulated in a dedicated, reusable Python module, and a unified command-line runner (`src/main.py`) enables end-to-end experimentation.

## 2. Algorithms

### 2.1 Dynamic Programming

**Module:** `src/agents/dp_agent.py`

Implements exact solutions assuming a known environment model:

- **Iterative Policy Evaluation** – Evaluates a fixed policy by computing its value function.
- **Policy Iteration** – Alternates between evaluation and greedy improvement until convergence.
- **Value Iteration** – Combines evaluation and improvement into a single sweep; extract the policy once converged.

All algorithms use the custom `MapEnv` grid-world (`src/envs/map_env.py`), which supports configurable grid size, obstacle density, and stochastic transitions.

### 2.2 Model-Free Methods

**Module:** `src/agents/model_free_agent.py`

Tabular algorithms that learn without a model:

- **Monte Carlo Policy Learning** – First-visit MC with epsilon-greedy exploration.
- **Q-Learning** – Off-policy TD(0) control.
- **SARSA** – On-policy TD(0) control.

These operate on Gymnasium discrete environments (e.g., `FrozenLake-v1`).

### 2.3 Deep Q-Networks

**Module:** `src/agents/dqn_agent.py`

Neural network–based value estimation with several stabilisation techniques:

| Feature | Toggle |
|---|---|
| Target Network (soft/hard update) | `use_target_network` |
| Double DQN | `use_double_dqn` |
| Dueling Architecture | `use_dueling` |

A replay buffer (`ReplayBuffer`) stores transitions for experience replay.

### 2.4 Policy Gradient (REINFORCE)

**Module:** `src/agents/pg_agent.py`

Direct policy optimisation via the REINFORCE algorithm:

- Stochastic policy parameterised by a neural network (`PolicyNetwork`).
- Optional value-function baseline (`ValueNetwork`) to reduce gradient variance.
- Discounted future rewards are normalised per episode.

### 2.5 Behavior Cloning

**Module:** `src/agents/bc_agent.py`

Supervised imitation learning from expert demonstrations:

- Cross-Entropy loss over discrete actions.
- Dataset-driven training loop with per-epoch shuffling.
- Save / load utilities for the learned policy.

## 3. Environment

### Custom Grid-World (`MapEnv`)
- Configurable N×N grid with random obstacles and a goal cell.
- Deterministic or stochastic transition dynamics.
- Visualisation via Matplotlib.

### Gymnasium Environments
- `FrozenLake-v1` (tabular)
- `LunarLander-v2` (continuous observations, discrete actions)

## 4. Results

Detailed training curves and evaluation metrics can be reproduced by running the CLI commands documented in the `README.md`. Animated GIF recordings of agent behaviour can be generated via the evaluation scripts.

## 5. Conclusion

This project demonstrates a progressive curriculum through RL: from guaranteed-optimal DP solutions, through sample-efficient tabular methods, to function approximation with deep networks and policy gradients, finally connecting to imitation learning via Behavior Cloning.
