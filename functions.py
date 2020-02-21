import numpy as np

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	value_function = np.zeros(nS)
	return value_function

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	new_policy = np.zeros(nS, dtype='int')
	return new_policy

def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	value_function = np.zeros(nS)
	policy = np.random.randint(low=0, high=4, size=nS, dtype=int)
	return value_function, policy

def value_iteration(P, nS, nA, gamma, tol):
	value_function = np.zeros(nS)
	value_function_prev = np.zeros(nS) + 1
	policy = np.zeros(nS, dtype='int')
	while(max(abs(value_function - value_function_prev)) > tol):#find optimal value function
		value_function_prev = np.copy(value_function)
		for s in range(nS):
			q_value = []
			for a in range(nA):
				next_states_rewards = []
				for prob, next_s, reward, done in P[s][a]:
					next_states_rewards.append((prob * (reward + gamma * value_function_prev[next_s])))
				q_value.append(sum(next_states_rewards))
			value_function[s] = max(q_value)
	for s in range(nS):#extract policy using optimal value function
		q_sa = np.zeros(nA)
		for a in range(nA):
			for prob, next_s, reward, done in P[s][a]:
				q_sa[a] += (prob * (reward + gamma * value_function[next_s]))
		policy[s] = np.argmax(q_sa)
	return value_function, policy