from hide_and_seek import environment
from hide_and_seek import agent

ob_size = 3
map_size = 9
max_steps = 32
num_episodes = 10000
epsilon = 1
alpha = 0.25
gamma = 0.95

agent = agent.dqn_agent(ob_size, load_weights=False)
for episode in range(num_episodes):
    print('epsilon =', epsilon)
    env = environment.env(map_size, ob_size)
    env.render(whole_map=True)
    for step in range(max_steps):
        ob = env.get_ob()
        action = agent.get_action(ob, epsilon)
        print(action)
        env.move(action)
        reward = env.get_reward()
        agent.give_reward(reward)
        if reward == 1:
            break
    print('value at t0 =', reward*gamma**step)
    agent.train_td(step + 1, alpha, gamma)
    agent.save()
    epsilon = epsilon - 0.001
    if epsilon < 0.1:
        epsilon = 0.1
    if episode % 25 == 0:
        agent.update_target_model()
