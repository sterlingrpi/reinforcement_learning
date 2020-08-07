from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense
import numpy as np

class dqn_agent:
    def __init__(self, seq_length, ob_size):
        model = Sequential()
        model.add(LSTM(seq_length, input_shape=(ob_size, ob_size)))
        model.add(BatchNormalization())
        model.add(Dense(4))
        self.dqn_model = model

        self.obs = np.zeros((seq_length, ob_size, ob_size))

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

    def train(self):
        return False
