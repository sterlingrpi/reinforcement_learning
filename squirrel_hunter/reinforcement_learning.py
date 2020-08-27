import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense, Flatten, TimeDistributed
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Input, Model
import numpy as np

class agent:
    def __init__(self, ob_shape, num_actions, load_weights=False, file_path='dqn_model.h5'):
        num_units = 32
        input = Input(shape=(1, ob_shape[0], ob_shape[1]))
        state = Input(shape=(6, num_units))
        x = TimeDistributed(Flatten())(input)
        x = Dense(num_units)(x)
        x, h1, c1 = LSTM(units=num_units, return_sequences=True, return_state=True)\
            (x, initial_state=[state[:, 0, :], state[:, 1, :]])
        x, h2, c2 = LSTM(units=num_units, return_sequences=True, return_state=True)\
            (x, initial_state=[state[:, 2, :], state[:, 3, :]])
        x, h3, c3 = LSTM(units=num_units, return_state=True)\
            (x, initial_state=[state[:, 4, :], state[:, 5, :]])
        new_state = tf.concat([tf.expand_dims(h1, axis=1),
                               tf.expand_dims(c1, axis=1),
                               tf.expand_dims(h2, axis=1),
                               tf.expand_dims(c2, axis=1),
                               tf.expand_dims(h3, axis=1),
                               tf.expand_dims(c3, axis=1)], axis=1)
        output = Dense(num_actions)(x)
        model = Model(inputs=[input, state], outputs=[output, new_state])
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        self.Q = model
        self.Q_target = model

        input = Input(shape=(1, ob_shape[0], ob_shape[1]))
        state = Input(shape=(6, num_units))
        x = TimeDistributed(Flatten())(input)
        x = Dense(num_units)(x)
        x, h1, c1 = LSTM(units=num_units, return_sequences=True, return_state=True)\
            (x, initial_state=[state[:, 0, :], state[:, 1, :]])
        x, h2, c2 = LSTM(units=num_units, return_sequences=True, return_state=True)\
            (x, initial_state=[state[:, 2, :], state[:, 3, :]])
        x, h3, c3 = LSTM(units=num_units, return_state=True)\
            (x, initial_state=[state[:, 4, :], state[:, 5, :]])
        output = Dense(num_actions)(x)
        model = Model(inputs=[input, state], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        self.Q_training = model

        if load_weights:
            self.Q.load_weights(file_path)
            self.Q_target.load_weights(file_path)
            self.Q_training.load_weights(file_path)

        self.ob_shape = ob_shape
        self.num_actions = num_actions

        self.obs = np.zeros(shape=(1, 1, self.ob_shape[0], self.ob_shape[1]), dtype=np.float32)
        self.vals = np.zeros(shape=(1, self.num_actions), dtype=np.float32)
        self.lstm_states = np.zeros(shape=(1, 6, num_units))
        self.actions = np.zeros(shape=1, dtype=np.int)
        self.rewards = np.zeros(shape=1)

    def reset_memory(self):
        self.obs = self.obs[None, -1]
        self.vals = self.vals[None, -1]
        self.lstm_states = self.lstm_states[None, -1]
        self.actions = self.actions[None, -1]
        self.rewards = self.rewards[None, -1]

    def update_memory(self, ob, vals, lstm_state, action):
        self.obs = np.append(self.obs, ob, axis=0)
        self.vals = np.append(self.vals, vals, axis=0)
        self.lstm_states = np.append(self.lstm_states, lstm_state, axis=0)
        self.actions = np.append(self.actions, [action], axis=0)

    def get_action(self, ob, epsilon):
        ob = np.expand_dims(ob, axis=0)
        vals, lstm_state = self.Q.predict((ob, self.lstm_states[None, -1]))
        action = int(np.argmax(vals))
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
        self.update_memory(ob, vals, lstm_state, action)
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
        t = np.random.randint(1, len(self.obs) - 1)
        ob_t = self.obs[None, t]
        ob_t_plus_1 = self.obs[None, t + 1]
        state_t = self.lstm_states[None, t - 1]
        #state_t_plus_1  = self.lstm_states[t]
        vals, state_t_plus_1 = self.Q_target.predict((ob_t, state_t))
        vals_t_plus_1, _ = self.Q_target.predict((ob_t_plus_1, state_t_plus_1))

        print('vals before =', vals)
        act_t = int(self.actions[t])
        vals[0, act_t] = self.rewards[t + 1] + gamma * np.amax(vals_t_plus_1)
        print('vals after =', vals)

        self.Q_training.fit((ob_t, state_t), vals, verbose=0)
        self.Q.set_weights(self.Q_training.get_weights())
        self.reset_memory()

    def train_batch(self, alpha, gamma):
        target_vals = np.copy(self.vals[:-1])
        for t in range(len(target_vals)):
            target_vals[t, self.actions[t]] = self.rewards[t] + gamma * np.amax(self.vals[t + 1])
        print(self.vals)
        print(target_vals)
        self.Q_training.fit((self.obs[:-1], self.lstm_states[:-1]), target_vals, verbose=0)
        self.Q.set_weights(self.Q_training.get_weights())
        self.reset_memory()


    def update_target_model(self):
        self.Q_target.set_weights(self.Q.get_weights())

    def save(self, file_path='dqn_model.h5'):
        self.Q.save(file_path)
