import numpy as np

class DPAgent:
    """
    Dynamic Programming Agent implementing Iterative Value Policy Evaluation,
    Policy Iteration, and Value Iteration algorithms on tabular grid environments.
    """
    
    @staticmethod
    def iterative_value_policy_evaluation(env: object, policy: np.ndarray, gamma: float=0.9, theta: float=0.1) -> np.ndarray:
        V = np.zeros(shape=(env.n_states,))
        while True:
            delta = 0
            for i in range(env.n_states):
                v = V[i]
                if not env.is_valid(i):
                    V[i] = -np.inf
                    continue
                else:
                    next_states, rewards, probability_trans = env.model(i, policy[i])
                    V_temp = 0
                    for j in range(len(next_states)):
                        V_temp += probability_trans[j] * (rewards[j] + gamma * V[next_states[j]])
                    V[i] = V_temp
                    delta = np.maximum(delta, np.abs(v - V[i]))

            if delta < theta:
                break
        return V

    @staticmethod
    def policy_iteration(env: object, gamma: float=0.9, theta: float=0.1):
        policy = np.random.randint(0, env.n_actions, size=(env.n_states,))
        while True:
            policy_prime = policy.copy()
            V = DPAgent.iterative_value_policy_evaluation(env, policy, gamma=gamma, theta=theta)
            for i in range(env.n_states):
                if not env.is_valid(i):
                    continue
                else:
                    V_max = -np.inf
                    for j in range(env.n_actions):
                        V_temp = 0
                        next_states, rewards, probability_trans = env.model(i, j)
                        for k in range(len(next_states)):
                            V_temp += probability_trans[k]*(rewards[k] + gamma*V[next_states[k]])
                        if(V_max < V_temp):
                            V_max = V_temp
                            policy[i] = j

            if (policy == policy_prime).all() == True:
                break
        return policy, V

    @staticmethod
    def value_iteration(env: object, gamma: float=0.9, theta: float=0.0001):
        V = np.zeros(shape=(env.n_states,))
        policy = np.zeros(shape=(env.n_states,), dtype=int)
        
        # 1. Compute optimal Value Function
        while True:
            delta = 0
            for i in range(env.n_states):
                v = V[i]
                if not env.is_valid(i):
                    V[i] = -np.inf
                    continue
                
                V_max = -np.inf
                for j in range(env.n_actions):
                    V_temp = 0
                    next_states, rewards, probability_trans = env.model(i, j)
                    for k in range(len(next_states)):
                        V_temp += probability_trans[k]*(rewards[k] + gamma*V[next_states[k]])
                    
                    if V_max < V_temp:
                        V_max = V_temp
                
                V[i] = V_max
                delta = np.maximum(delta, np.abs(v - V[i]))
                
            if delta < theta:
                break
                
        # 2. Extract optimal policy
        for i in range(env.n_states):
            if env.is_valid(i) == False:
                continue
            
            V_max = -np.inf
            best_action = 0
            for j in range(env.n_actions):
                V_temp = 0
                next_states, rewards, probability_trans = env.model(i, j)
                for k in range(len(next_states)):
                    V_temp += probability_trans[k]*(rewards[k] + gamma*V[next_states[k]])
                
                if V_max < V_temp:
                    V_max = V_temp
                    best_action = j
            
            policy[i] = best_action

        return policy, V
