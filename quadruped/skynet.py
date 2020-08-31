import tensorflow as tf
import paramiko
import pickle
import numpy as np
from tensorflow.keras import models
import time

class agent:
    def __init__(self):
        #initialize sftp
        self.data_file_name = 'data.pickle'
        self.model_file_name = 'Q_new.tflite'
        self.source = r'D:/PycharmProjects/reinforcement_learning/quadruped/'
        self.dest = r'/home/mendel/birdie/'
        hostname = '192.168.0.23'
        port = 22  # default port for SSH
        username = 'mendel'
        password = 'mendel'
        t = paramiko.Transport((hostname, port))
        t.connect(username=username, password=password)
        self.sftp = paramiko.SFTPClient.from_transport(t)

        #initialize Q model
        self.file_path = 'D:/PycharmProjects/reinforcement_learning/squirrel_hunter/Q.h5'
        self.Q = models.load_model(self.file_path)
        self.Q.load_weights(self.file_path)
        self.Q_training = models.load_model('D:/PycharmProjects/reinforcement_learning/squirrel_hunter/Q_training.h5')
        self.Q_training.set_weights(self.Q.get_weights())

    def check_for_data(self):
        try:
            self.sftp.get(self.dest + self.data_file_name, self.source + self.data_file_name)
            self.sftp.remove(self.dest + self.data_file_name)
            return True
        except:
            return False

    def ship_new_model(self):
        for i in range(10):
            try:
                self.sftp.put(self.source + self.model_file_name, self.dest + self.model_file_name)
                break
            except:
                pass

    def train(self, gamma = 0.95):
        with open(self.data_file_name, 'rb') as f:
            data = pickle.load(f)
        actions, lstm_states, obs, rewards, vals = data
        if len(actions) >= 2:
            target_vals = np.copy(vals[:-1])
            for t in range(len(target_vals)):
                target_vals[t, actions[t]] = rewards[t] + gamma * np.amax(vals[t + 1])
            self.Q_training.fit((obs[:-1], lstm_states[:-1]), target_vals, verbose=0)
            self.Q.set_weights(self.Q_training.get_weights())
            self.Q.save(self.file_path)

    def convert_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.Q)
        tflite_model = converter.convert()
        open('./Q_new.tflite', "wb").write(tflite_model)

if __name__ == '__main__':
    genisys = agent()
    while True:
        if genisys.check_for_data():
            print('training')
            genisys.train()
            genisys.convert_tflite()
            genisys.ship_new_model()
        else:
            print('sleeping', time.time())
            time.sleep(5)
