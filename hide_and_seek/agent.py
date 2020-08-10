import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Flatten, TimeDistributed
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

class dqn_agent:
    def __init__(self, ob_size, load_weights=False, file_path='dqn_model.h5'):
        input = Input(shape=(None, ob_size, ob_size))
        x = TimeDistributed(Flatten())(input)
        x = LSTM(units=10)(x)
        x = BatchNormalization()(x)
        output = Dense(4)(x)
        model = Model(input, output)
        model.compile(loss='mse',optimizer=Adam(lr=0.1))
        self.dqn_model = model
        if load_weights:
            self.dqn_model.load_weights(file_path)

        self.ob_size = ob_size
        self.obs = np.zeros(shape=(1, self.ob_size, self.ob_size), dtype=np.float32)
        self.policies = np.zeros(shape=1)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.1, nesterov=False, name='SGD')

    def get_action(self, ob, epsilon):
        self.obs = np.append(self.obs, [ob], axis=0)
        if np.random.random() < epsilon:
            policy = np.random.randint(0, 4)
        else:
            policy = np.argmax(self.dqn_model(np.array([self.obs[1:]])))
        self.policies = np.append(self.policies, [policy], axis=0)
        if policy == 0:
            action = 'w'
        elif policy == 1:
            action = 'a'
        elif policy == 2:
            action = 's'
        elif policy == 3:
            action = 'd'
        return action

    def train(self, num_steps, value, gamma):
        for t in range(num_steps):
            obs_t = np.array([self.obs[1:t + 2]])
            Q = self.dqn_model.predict(obs_t)
            action_t = int(self.policies[t + 1])
            Q[0, action_t] = value*gamma**(num_steps - t -1)
            self.dqn_model.fit(obs_t, Q, verbose=0)
        self.obs = np.zeros(shape=(1, self.ob_size, self.ob_size))
        self.policies = np.zeros(shape=1)

    def save(self, file_path='dqn_model.h5'):
        self.dqn_model.save(file_path)
