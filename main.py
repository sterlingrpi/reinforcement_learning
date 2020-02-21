import numpy as np
import gym
import time
from lake_envs import *
from functions import *

np.set_printoptions(precision=3)

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
	#render_single(env, p_pi, max_steps=10)
	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, max_steps=10)
