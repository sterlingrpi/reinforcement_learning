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
	value_function = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
	policy = np.random.randint(low=0, high=4, size=nS, dtype=int)
	return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	value_function = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
	policy = np.random.randint(low=0, high=4, size=nS, dtype=int)
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
	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, max_steps=10)
	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, max_steps=10)
