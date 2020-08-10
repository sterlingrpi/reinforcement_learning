from hide_and_seek import environment
from hide_and_seek import agent

ob_size = 3
map_size = 6
max_steps = 32
num_episodes = 1000
gamma = 0.95
epsilon = 0.25

agent = agent.dqn_agent(ob_size, load_weights=False)
for episode in range(num_episodes):
    env = environment.env(map_size, ob_size)
    for step in range(max_steps):
        ob = env.get_ob()
        action = agent.get_action(ob, epsilon)
        env.move(action)
        value = env.get_value()
        if value == 1:
            break

    reward = value*gamma**step
    print('reward =', reward)
    agent.train(step + 1, value, gamma)
    agent.save()
