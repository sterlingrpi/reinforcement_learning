import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense
import numpy as np

class dqn_agent:
    def __init__(self, seq_length, ob_size, load_weights=False, file_path='dqn_model.h5'):
        model = Sequential()
        model.add(LSTM(seq_length, input_shape=(ob_size, ob_size)))
        model.add(BatchNormalization())
        model.add(Dense(4))
        self.dqn_model = model
        if load_weights:
            self.dqn_model.load_weights(file_path)

        self.obs = np.zeros((seq_length, ob_size, ob_size))

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')

    def get_action(self, ob, epsilon):
        self.obs = np.concatenate(([ob], self.obs[1:]))
        if np.random.random() < epsilon:
            prediction = np.random.randint(0, 3)
        else:
            prediction = np.argmax(self.dqn_model.predict(self.obs))
        if prediction == 0:
            action = 'w'
        elif prediction == 1:
            action = 'a'
        elif prediction == 2:
            action = 's'
        elif prediction == 3:
            action = 'd'
        return action

    def train(self, reward):
        with tf.GradientTape() as tape:
            predictions = self.dqn_model(tf.constant(self.obs))
            loss = (1-reward)*predictions
        variables = self.dqn_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def save(self, file_path='dqn_model.h5'):
        self.dqn_model.save(file_path)
