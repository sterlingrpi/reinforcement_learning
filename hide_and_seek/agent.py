from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Flatten, TimeDistributed
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

class dqn_agent:
    def __init__(self, ob_size, load_weights=False, file_path='dqn_model.h5'):
        input = Input(shape=(None, ob_size, ob_size))
        x = TimeDistributed(Flatten())(input)
        x = LSTM(units=10)(x)
        output = Dense(4)(x)
        model = Model(input, output)
        model.compile(loss='mse',optimizer=Adam(lr=0.01))

        self.Q = model
        self.Q_target = model
        if load_weights:
            self.Q.load_weights(file_path)
            self.Q_target.load_weights(file_path)

        self.ob_size = ob_size
        self.reset_memory()

    def reset_memory(self):
        self.obs = np.zeros(shape=(1, self.ob_size, self.ob_size), dtype=np.float32)
        self.actions = np.zeros(shape=1)
        self.rewards = np.zeros(shape=1)

    def get_action(self, ob, epsilon):
        self.obs = np.append(self.obs, [ob], axis=0)
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(self.Q(np.array([self.obs[1:]])))
        self.actions = np.append(self.actions, [action], axis=0)
        if action == 0:
            action_wasd = 'w'
        elif action == 1:
            action_wasd = 'a'
        elif action == 2:
            action_wasd = 's'
        elif action == 3:
            action_wasd = 'd'
        return action_wasd

    def give_reward(self, reward):
        self.rewards = np.append(self.rewards, [reward], axis=0)

    def train_td(self, num_steps, alpha, gamma):
        for t in range(num_steps):
            t += 1
            obs_t = np.array([self.obs[1:t + 1]])
            vals = self.Q.predict(obs_t)
            act_t = int(self.actions[t])
            if t == num_steps:
                vals[0, act_t] = self.rewards[t]
            else:
                obs_t_plus_1 = np.array([self.obs[1:t + 2]])
                vals_target = self.Q_target.predict(obs_t_plus_1)
                vals[0, act_t] = vals[0, act_t] + alpha*(self.rewards[t] + gamma*np.amax(vals_target) - vals[0, act_t])
            self.Q.fit(obs_t, vals, verbose=0)
        self.reset_memory()

    def train_monte_carlo(self, num_steps, alpha, gamma):
        for t in range(num_steps):
            t += 1
            obs_t = np.array([self.obs[1:t + 1]])
            vals = self.Q_target.predict(obs_t)
            act_t = int(self.actions[t])
            vals[0, act_t] = self.rewards[-1]*gamma**(num_steps - t)
            self.Q.fit(obs_t, vals, verbose=0)
        self.reset_memory()

    def save(self, file_path='dqn_model.h5'):
        self.Q.save(file_path)

    def update_target_model(self):
        self.Q_target.set_weights(self.Q.get_weights())