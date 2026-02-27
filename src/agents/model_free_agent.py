import numpy as np

class ModelFreeAgent:
    """
    Model-Free Reinforcement Learning Agent implementing Monte Carlo, 
    Q-Learning, and SARSA algorithms for tabular environments (e.g. FrozenLake).
    """
    
    @staticmethod
    def epsilon_greedy_policy(Q, state, epsilon, n_actions):
        if np.random.rand() < (1 - epsilon):
            action = np.argmax(Q[state, :])
        else:
            action = np.random.choice(range(n_actions))
        return action

    @staticmethod
    def generate_trial_using_epsilon_greedy_policy(env, policy, epsilon=0.0):
        states = []
        actions = []
        rewards = []

        obs, _ = env.reset()
        state = obs
        states.append(state)
        while True:
            if np.random.rand() < (1 - epsilon):
                action = policy[state]
            else:
                action = np.random.choice([a for a in range(env.action_space.n) if a != policy[state]])
            
            actions.append(int(action))
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            states.append(state)

            if terminated or truncated:
                break
        
        return states, actions, rewards

    @staticmethod
    def monte_carlo_policy_learning(env, epsilon=0.1, gamma=0.9, alpha=0.1, episodes=1000):
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        policy = np.random.randint(env.action_space.n, size=env.observation_space.n)
        for i in range(episodes):
            states, actions, rewards = ModelFreeAgent.generate_trial_using_epsilon_greedy_policy(env, policy, epsilon)
            G = 0
            for t in reversed(range(len(states) - 1)):
                G = rewards[t] + gamma * G
                pair_s_a = [(states[j], actions[j]) for j in range(t)]
                if (states[t], actions[t]) not in pair_s_a:
                    Q[states[t], actions[t]] += alpha * (G - Q[states[t], actions[t]])
                    policy[states[t]] = np.argmax(Q[states[t], :])
               
        return Q, policy

    @staticmethod
    def monte_carlo_policy_evaluation(env, policy, gamma=0.9, alpha=0.1, episodes=1000):
        V = np.zeros(env.observation_space.n)
        for i in range(episodes):
            states, actions, rewards = ModelFreeAgent.generate_trial_using_epsilon_greedy_policy(env, policy, epsilon=0.0)
            G = 0
            for t in reversed(range(len(states) - 1)):
                G = rewards[t] + gamma * G
                s = states[t]
                if s not in [st for st in states[0:t]]:
                    V[s] += alpha * (G - V[s])
        return V

    @staticmethod
    def q_learning(env, episodes, alpha=0.1, gamma=0.9, epsilon=0.3):
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        policy = np.zeros(env.observation_space.n, dtype=int)
        for i in range(episodes):
            obs, _ = env.reset()
            state = obs
            while True:
                action = ModelFreeAgent.epsilon_greedy_policy(Q, state, epsilon, env.action_space.n)
                state_prime, reward, terminated, truncated, info = env.step(action)
                Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[state_prime, :]) - Q[state, action])
                policy[state] = np.argmax(Q[state, :])
                state = state_prime

                if terminated or truncated:
                    break
       
        return Q, policy

    @staticmethod
    def sarsa(env, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        policy = np.zeros(env.observation_space.n, dtype=int)
        for i in range(episodes):
            obs, _ = env.reset()
            state = obs
            action = ModelFreeAgent.epsilon_greedy_policy(Q, state, epsilon, env.action_space.n)
            while True:
                state_prime, reward, terminated, truncated, info = env.step(action)
                action_prime = ModelFreeAgent.epsilon_greedy_policy(Q, state_prime, epsilon, env.action_space.n)
                Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[state_prime, action_prime] - Q[state, action])
                state = state_prime
                action = action_prime

                if terminated or truncated:
                    break
        
        for s in range(env.observation_space.n):
            policy[s] = np.argmax(Q[s, :])

        return Q, policy
