"""
Unified Pipeline Runner for Reinforcement Learning Labs.

Provides CLI-based entry-points for running each RL algorithm
implemented in this project.

Usage examples:
    python src/main.py dp --algorithm value_iteration
    python src/main.py model_free --algorithm q_learning --episodes 5000
    python src/main.py dqn --episodes 500
    python src/main.py pg --episodes 2000
    python src/main.py bc --epochs 100
"""

import argparse
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Dynamic Programming
# ---------------------------------------------------------------------------

def run_dp(args):
    from envs.map_env import MapEnv
    from agents.dp_agent import DPAgent
    from utils.rl_aux import print_policy, print_value_function

    env = MapEnv(size=args.size, obstacles_percent=args.obstacles,
                 stochastic=args.stochastic)
    env.plot()

    gamma = args.gamma
    theta = args.theta
    algorithm = args.algorithm

    if algorithm == "policy_evaluation":
        policy = np.random.randint(0, env.n_actions, size=(env.n_states,))
        V = DPAgent.iterative_value_policy_evaluation(env, policy,
                                                      gamma=gamma, theta=theta)
        print_value_function(env, V)
        print_policy(env, policy)

    elif algorithm == "policy_iteration":
        policy, V = DPAgent.policy_iteration(env, gamma=gamma, theta=theta)
        print_value_function(env, V)
        print_policy(env, policy)

    elif algorithm == "value_iteration":
        policy, V = DPAgent.value_iteration(env, gamma=gamma, theta=theta)
        print_value_function(env, V)
        print_policy(env, policy)

    else:
        print(f"Unknown DP algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Model-Free (MC, Q-learning, SARSA)
# ---------------------------------------------------------------------------

def run_model_free(args):
    import gymnasium as gym
    from agents.model_free_agent import ModelFreeAgent

    env = gym.make(args.env)
    agent = ModelFreeAgent(env)

    algorithm = args.algorithm
    episodes = args.episodes
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon

    if algorithm == "q_learning":
        Q, policy = agent.q_learning(episodes, alpha=alpha, gamma=gamma,
                                     epsilon=epsilon)
    elif algorithm == "sarsa":
        Q, policy = agent.sarsa(episodes, alpha=alpha, gamma=gamma,
                                epsilon=epsilon)
    elif algorithm == "monte_carlo":
        Q, policy = agent.monte_carlo_policy_learning(
            episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)
    else:
        print(f"Unknown model-free algorithm: {algorithm}")
        return

    print(f"Training complete after {episodes} episodes.")
    print(f"Policy: {policy}")


# ---------------------------------------------------------------------------
# Deep Q-Network (DQN / DDQN / Dueling)
# ---------------------------------------------------------------------------

def run_dqn(args):
    import gymnasium as gym
    from agents.dqn_agent import DQNAgent

    env = gym.make(args.env)
    agent = DQNAgent(
        gamma=args.gamma, epsilon=1.0, lr=args.lr,
        input_dims=env.observation_space.shape[0],
        batch_size=args.batch_size,
        n_actions=env.action_space.n,
        eps_end=args.eps_end, eps_dec=args.eps_dec,
        use_target_network=True,
        use_double_dqn=args.double,
        use_dueling=args.dueling,
    )

    scores = []
    n_games = args.episodes
    for i in range(n_games):
        obs, _ = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.store_transition(obs, action, reward,
                                                 new_obs, done)
            agent.learn()
            score += reward
            obs = new_obs
        scores.append(score)

        if (i + 1) % 50 == 0:
            avg = np.mean(scores[-50:])
            print(f"episode {i}  score {score:.2f}  "
                  f"average score {avg:.2f}  epsilon {agent.epsilon:.2f}")

    env.close()
    print("DQN training complete.")


# ---------------------------------------------------------------------------
# Policy Gradient (REINFORCE)
# ---------------------------------------------------------------------------

def run_pg(args):
    import gymnasium as gym
    from agents.pg_agent import VPGAgent

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    agent = VPGAgent(obs_dim, n_acts, hidden_layers_dim=args.hidden,
                     lr=args.lr, gamma=args.gamma,
                     use_baseline=args.baseline)

    score_history = []
    n_episodes = args.episodes

    for i in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        rewards = []
        logprobs = []
        states = []

        while not done:
            action, logprob = agent.get_action(obs)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            logprobs.append(logprob)
            states.append(obs)
            total_reward += reward
            obs = new_obs

        score_history.append(total_reward)
        fut_rew = agent.discounted_future_reward(rewards)
        agent.learn(fut_rew, logprobs,
                    states=states if args.baseline else None)

        if (i + 1) % 50 == 0:
            avg = np.mean(score_history[-50:])
            print(f"Episode {i}  total_reward: {total_reward:.2f}, "
                  f"average reward {avg:.2f}")

    env.close()
    print("Policy Gradient training complete.")


# ---------------------------------------------------------------------------
# Behavior Cloning
# ---------------------------------------------------------------------------

def run_bc(args):
    import gymnasium as gym
    from agents.bc_agent import BCAgent

    dataset_path = args.dataset
    loaded = np.load(dataset_path)
    dataset = dict(loaded)

    agent = BCAgent(
        input_dims=dataset['observations'].shape[1],
        n_actions=int(dataset['actions'].max()) + 1,
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_layers=args.hidden,
    )
    agent.learn(verbose=True)

    # Evaluate
    env = gym.make(args.env, render_mode="rgb_array")
    total_rewards = []
    for _ in range(args.eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)

    env.close()
    print(f"BC Evaluation over {args.eval_episodes} episodes: "
          f"Mean reward = {np.mean(total_rewards):.2f} "
          f"+/- {np.std(total_rewards):.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning Labs – Unified Runner")
    subparsers = parser.add_subparsers(dest="command")

    # --- DP ---
    dp_p = subparsers.add_parser("dp", help="Dynamic Programming algorithms")
    dp_p.add_argument("--algorithm", type=str, default="value_iteration",
                      choices=["policy_evaluation", "policy_iteration",
                               "value_iteration"])
    dp_p.add_argument("--size", type=int, default=4)
    dp_p.add_argument("--obstacles", type=float, default=0.2)
    dp_p.add_argument("--stochastic", action="store_true")
    dp_p.add_argument("--gamma", type=float, default=0.9)
    dp_p.add_argument("--theta", type=float, default=0.0001)

    # --- Model-Free ---
    mf_p = subparsers.add_parser("model_free",
                                 help="Model-Free RL (MC, Q-learning, SARSA)")
    mf_p.add_argument("--algorithm", type=str, default="q_learning",
                      choices=["q_learning", "sarsa", "monte_carlo"])
    mf_p.add_argument("--env", type=str, default="FrozenLake-v1")
    mf_p.add_argument("--episodes", type=int, default=10000)
    mf_p.add_argument("--alpha", type=float, default=0.1)
    mf_p.add_argument("--gamma", type=float, default=0.9)
    mf_p.add_argument("--epsilon", type=float, default=0.3)

    # --- DQN ---
    dqn_p = subparsers.add_parser("dqn", help="Deep Q-Network variants")
    dqn_p.add_argument("--env", type=str, default="LunarLander-v2")
    dqn_p.add_argument("--episodes", type=int, default=500)
    dqn_p.add_argument("--lr", type=float, default=0.001)
    dqn_p.add_argument("--gamma", type=float, default=0.99)
    dqn_p.add_argument("--batch_size", type=int, default=64)
    dqn_p.add_argument("--eps_end", type=float, default=0.02)
    dqn_p.add_argument("--eps_dec", type=float, default=5e-5)
    dqn_p.add_argument("--double", action="store_true",
                       help="Use Double DQN")
    dqn_p.add_argument("--dueling", action="store_true",
                       help="Use Dueling DQN architecture")

    # --- PG ---
    pg_p = subparsers.add_parser("pg",
                                 help="Policy Gradient (REINFORCE)")
    pg_p.add_argument("--env", type=str, default="LunarLander-v2")
    pg_p.add_argument("--episodes", type=int, default=2000)
    pg_p.add_argument("--lr", type=float, default=0.005)
    pg_p.add_argument("--gamma", type=float, default=0.99)
    pg_p.add_argument("--hidden", type=int, default=128)
    pg_p.add_argument("--baseline", action="store_true",
                      help="Use value-function baseline")

    # --- BC ---
    bc_p = subparsers.add_parser("bc", help="Behavior Cloning")
    bc_p.add_argument("--env", type=str, default="LunarLander-v2")
    bc_p.add_argument("--dataset", type=str,
                      default="data/datasets/lunar_lander_100.npz")
    bc_p.add_argument("--epochs", type=int, default=100)
    bc_p.add_argument("--batch_size", type=int, default=64)
    bc_p.add_argument("--lr", type=float, default=0.003)
    bc_p.add_argument("--hidden", type=int, default=64)
    bc_p.add_argument("--eval_episodes", type=int, default=20)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "dp": run_dp,
        "model_free": run_model_free,
        "dqn": run_dqn,
        "pg": run_pg,
        "bc": run_bc,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
