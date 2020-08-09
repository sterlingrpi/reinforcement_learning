import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras import Input, Model
import numpy as np

class dqn_agent:
    def __init__(self, ob_size, load_weights=False, file_path='dqn_model.h5'):
        input = Input(shape=(None, 9))
        x = LSTM(units=32)(input)
        x = BatchNormalization()(x)
        output = Dense(4)(x)
        model = Model(input, output)
        self.dqn_model = model
        if load_weights:
            self.dqn_model.load_weights(file_path)
        self.ob_size = ob_size
        self.obs = np.zeros(shape=(1, self.ob_size, self.ob_size))
        self.policies = np.zeros(shape=1)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')

    def get_action(self, ob, epsilon):
        self.obs = np.append([ob], self.obs, axis=0)
        if np.random.random() < epsilon:
            policy = np.random.randint(0, 4)
        else:
            flattened_obs = np.reshape(self.obs, newshape=(1, self.obs.shape[0], self.obs.shape[1]*self.obs.shape[2]))[:, :self.obs.shape[0]-1, :]
            policy = np.argmax(self.dqn_model.predict(flattened_obs))
        self.policies = np.append([policy], self.policies, axis=0)
        if policy == 0:
            action = 'w'
        elif policy == 1:
            action = 'a'
        elif policy == 2:
            action = 's'
        elif policy == 3:
            action = 'd'
        return action

    def train(self, reward, num_steps):
        for step in range(num_steps):
            flattened_obs = np.reshape(self.obs, newshape=(1, self.obs.shape[0], self.obs.shape[1] * self.obs.shape[2]))[:,:step + 1, :]
            target_policy = tf.reshape(tf.one_hot(self.policies[step + 1], depth=4), shape=(1, 4))
            with tf.GradientTape() as tape:
                policy = self.dqn_model(flattened_obs)
                loss = (target_policy - policy**2)*reward - 0.05
            #print(policy)
            #print(loss)
            #loss = tf.clip_by_value(loss, 0, 1)
            variables = self.dqn_model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
        self.obs = np.zeros(shape=(1, self.ob_size, self.ob_size))
        self.policies = np.zeros(shape=1)

    def save(self, file_path='dqn_model.h5'):
        self.dqn_model.save(file_path)
