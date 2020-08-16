import numpy as np

class env:
    def __init__(self, map_size=9, ob_size=3):
        #self.map = np.random.randint(0, 7, (map_size, map_size))
        self.map = np.array([[1, 1, 1, 0, 0, 0, 2, 2, 2],
                             [1, 1, 1, 0, 0, 0, 2, 2, 2],
                             [1, 1, 1, 0, 0, 0, 2, 2, 2],
                             [0, 0, 0, 3, 3, 3, 0, 0, 0],
                             [0, 0, 0, 3, 3, 3, 0, 0, 0],
                             [0, 0, 0, 3, 3, 3, 0, 0, 0],
                             [4, 4, 4, 0, 0, 0, 5, 5, 5],
                             [4, 4, 4, 0, 0, 0, 5, 5, 5],
                             [4, 4, 4, 0, 0, 0, 5, 5, 5]])
        x = np.random.randint(0, map_size)
        y = np.random.randint(0, map_size)
        self.map[x, y] = 7
        self.fov = np.array([0, ob_size, 0, ob_size]) #+ np.random.randint(map_size - ob_size + 1)

    def move(self, direction):
        new_fov = np.copy(self.fov)
        if direction == 'w':
            new_fov[:2] = self.fov[:2]-1
        elif direction == 'a':
            new_fov[2:] = self.fov[2:]-1
        elif direction == 'd':
            new_fov[2:] = self.fov[2:]+1
        elif direction == 's':
            new_fov[:2] = self.fov[:2]+1
        if np.amin(new_fov) >= 0 and np.amax(new_fov) <= self.map.shape[0]:
            self.fov = new_fov

    def get_ob(self):
        return self.map[self.fov[0]:self.fov[1], self.fov[2]:self.fov[3]]

    def render(self, whole_map=False):
        if whole_map:
            ob = self.map
        else:
            ob = self.get_ob()
        for row in ob:
            print(row)

    def get_reward(self):
        ob = self.get_ob()
        if 7 in ob:
            reward = 1
        else:
            reward = 0
        return reward