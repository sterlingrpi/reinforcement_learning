from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense, Flatten, TimeDistributed
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Input, Model
import numpy as np

class agent:
    def __init__(self, ob_shape, num_actions, load_weights=False, file_path='dqn_model.h5'):
        input = Input(batch_shape=(1, 1, ob_shape[0], ob_shape[1]))
        x = TimeDistributed(Flatten())(input)
        x = Dense(32)(x)
        x = LSTM(units=32, return_sequences=True, stateful=True)(x)
        x = LSTM(units=32, return_sequences=True, stateful=True)(x)
        x = LSTM(units=32, stateful=True)(x)
        output = Dense(num_actions)(x)
        model = Model(input, output)
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        self.Q_stateful = model

        input = Input(shape=(None, ob_shape[0], ob_shape[1]))
        x = TimeDistributed(Flatten())(input)
        x = Dense(32)(x)
        x = LSTM(units=32, return_sequences=True)(x)
        x = LSTM(units=32, return_sequences=True)(x)
        x = LSTM(units=32)(x)
        output = Dense(num_actions)(x)
        model = Model(input, output)
        model.compile(loss='mse',optimizer=Adam(lr=0.01))
        self.Q = model
        self.Q_target = model

        if load_weights:
            self.Q_stateful.load_weights(file_path)
            self.Q.load_weights(file_path)
            self.Q_target.load_weights(file_path)

        self.ob_shape = ob_shape
        self.reset_memory()

    def reset_memory(self):
        self.obs = np.zeros(shape=(1, self.ob_shape[0], self.ob_shape[1]), dtype=np.float32)
        self.actions = np.zeros(shape=1)
        self.rewards = np.zeros(shape=1)
        self.Q_stateful.reset_states()

    def get_action(self, ob, epsilon):
        self.obs = np.append(self.obs, ob, axis=0)
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

    def train(self, alpha, gamma):
        print('shape obs =', self.obs.shape)
        if self.obs.shape[0] >= 4:
            obs_t = np.array([self.obs[1:-1]])
            obs_t_plus_1 = np.array([self.obs[1:]])
            vals = self.Q_target.predict(obs_t)
            vals_target = self.Q_target.predict(obs_t_plus_1)

            print('vals before =', vals)
            act_t = int(self.actions[-1])
            vals[0, act_t] = vals[0, act_t] + alpha*(self.rewards[-1] + gamma*np.amax(vals_target) - vals[0, act_t])
            #vals[0, act_t] = self.rewards[-1] + gamma * np.amax(vals_target)
            print('vals after =', vals)

            self.Q.fit(obs_t, vals, verbose=0)
            self.Q_stateful.set_weights(self.Q.get_weights())
            self.reset_memory()

    def update_target_model(self):
        self.Q_target.set_weights(self.Q.get_weights())

    def save(self, file_path='dqn_model.h5'):
        self.Q.save(file_path)
