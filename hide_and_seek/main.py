from hide_and_seek import environment
from hide_and_seek import agent

ob_size = 3
map_size = 9
max_steps = 32
gamma = 0.95
epsilon = 0.2

agent = agent.dqn_agent(max_steps, ob_size)
env = environment.env(map_size, ob_size)
env.render(whole_map=True)

for step in range(max_steps):
    env.move(input())
    env.render()

    ob = env.get_ob()
    action = agent.get_action(ob, epsilon)
    print('agent action:', action)

    value = env.get_value()
    print('value =', value)
    if value == 1:
        break

reward = value*gamma**step
print('reward =', reward)
