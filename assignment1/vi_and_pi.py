import numpy as np
import gym
import time
from assignment1.lake_envs import *

np.set_printoptions(precision=3)

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
	policy = np.random.randint(low=0, high=4, size=nS, dtype=int)
	for i in range(1000):#find optimal value function
		value_function_prev = value_function
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

def render_single(env, policy, max_steps):
  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)

if __name__ == "__main__":
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print('env.P =', env.P)
	print('env.P[s][a] =', env.P[0][0])
	print('env.nS =', env.nS)
	print('env.nA =', env.nA)
	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, max_steps=10)
	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, max_steps=10)
