import tflite_runtime.interpreter as tflite
import platform
import cv2
import numpy as np
from move import dove_move_tripod, dove_move_diagonal, robot_hight, robot_stand
import pickle
import time
import os

class agent:
    def __init__(self):
        #initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        #initialize image segmentation
        EDGETPU_SHARED_LIB = {'Linux': 'libedgetpu.so.1', 'Darwin': 'libedgetpu.1.dylib', 'Windows': 'edgetpu.dll'}[platform.system()]
        self.interpreter_seg = tflite.Interpreter(model_path='seg_512_128_edgetpu.tflite', experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB, {})])
        self.interpreter_seg.allocate_tensors()
        self.input_details_seg = self.interpreter_seg.get_input_details()
        self.output_details_seg = self.interpreter_seg.get_output_details()
        self.image_size = (self.input_details_seg[0]['shape'][2], self.input_details_seg[0]['shape'][1])
        self.num_classes = self.output_details_seg[0]['shape'][3]

        #initialize DQN
        self.Q = tflite.Interpreter(model_path='Q.tflite')
        self.check_for_new_model()
        self.Q.allocate_tensors()
        self.input_details_dqn = self.Q.get_input_details()
        self.output_details_dqn = self.Q.get_output_details()
        self.directions = ['forward', 'left','backward', 'right']

        #initialize memory
        self.actions = np.zeros(shape=1, dtype=np.int)
        self.lstm_states = np.zeros(shape=(1, 6, 64), dtype=np.float32)
        self.obs = np.zeros(shape=(1, 1, 128, 128), dtype=np.float32)
        self.rewards = np.zeros(shape=1)
        self.vals = np.zeros(shape=(1, 4), dtype=np.float32)
        self.times_since_last_save = 0

    def update_memory(self, action, lstm_state, ob, reward, vals):
        self.actions = np.append(self.actions, [action], axis=0)
        self.lstm_states = np.append(self.lstm_states, lstm_state, axis=0)
        self.obs = np.append(self.obs, ob, axis=0)
        self.rewards = np.append(self.rewards, [reward], axis=0)
        self.vals = np.append(self.vals, vals, axis=0)

    def reset_memory(self):
        self.actions = self.actions[None, -1]
        self.lstm_states = self.lstm_states[None, -1]
        self.obs = self.obs[None, -1]
        self.rewards = self.rewards[None, -1]
        self.vals = self.vals[None, -1]

    def get_ob(self):
        return_value, image = self.camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA))/255.0
        image = [image.astype('float32')]
        self.interpreter_seg.set_tensor(self.input_details_seg[0]['index'], image)
        self.interpreter_seg.invoke()
        ob = self.interpreter_seg.get_tensor(self.output_details_seg[0]['index'])
        ob = np.argmax(ob, axis=3)
        reward = 0
        if np.count_nonzero(ob == 3) > 100:
            reward = -1
        elif np.count_nonzero(ob == 9) > 100:
            reward = 1
        return np.expand_dims(ob, axis=0).astype(np.float32), reward

    def get_action(self, ob, reward, epsilon):
        self.Q.set_tensor(self.input_details_dqn[0]['index'], ob)
        self.Q.set_tensor(self.input_details_dqn[1]['index'], self.lstm_states[None, -1])
        self.Q.invoke()
        vals = self.Q.get_tensor(self.output_details_dqn[0]['index'])
        lstm_state = self.Q.get_tensor(self.output_details_dqn[1]['index'])
        action = int(np.argmax(vals))
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
        self.update_memory(action, lstm_state, ob, reward, vals)
        return action

    def do_move(self, action):
        direction = self.directions[action]
        for step_input in range(9):
            step_input += 1
            # dove_move_tripod(step_input, 50, direction)
            dove_move_diagonal(step_input, 100, direction)

    def save_memory_to_file(self, force_save=False):
        if self.times_since_last_save > 25 or force_save:
            self.times_since_last_save = 0
            data = (self.actions, self.lstm_states, self.obs, self.rewards, self.vals)
            with open('data.pickle', 'wb') as f:
                pickle.dump(data, f)
            self.reset_memory()
        self.times_since_last_save += 1

    def check_for_new_model(self):
        try:
            self.Q = tflite.Interpreter(model_path='Q_new.tflite')
            os.remove('Q.tflite')
            os.rename('Q_new.tflite', 'Q.tflite')
            print('new Q model loaded')
        except:
            pass

    def should_attack(self, ob):
        return np.count_nonzero(ob == 3) > 100 or np.count_nonzero(ob == 9) < 75 and np.count_nonzero(ob == 2) < 75


if __name__ == '__main__':
    T800 = agent()
    state = 'patrol'
    while True:
        if state == 'patrol':
            print('patrolling:', time.time())
            ob, reward = T800.get_ob()
            if T800.should_attack(ob):
                state = 'attack'
            else:
                T800.check_for_new_model()
                time.sleep(1)

        if state == 'attack':
            print('attacking:', time.time())
            ob, reward = T800.get_ob()
            if T800.should_attack(ob):
                action = T800.get_action(ob, reward, epsilon=0)
                T800.do_move(action)
                T800.save_memory_to_file()
            else:
                state = 'patrol'
                T800.save_memory_to_file(force_save=True)
