import numpy as np

class env:
    def __init__(self, map_size=9, ob_size=3):
        self.map = np.zeros((map_size, map_size))
        x = np.random.randint(0, map_size)
        if x > ob_size:
            y = np.random.randint(0, map_size)
        else:
            y = np.random.randint(ob_size, map_size)
        self.map[x, y] = 1
        self.fov = np.array([0, ob_size, 0, ob_size])# + map_size//2

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

    def get_value(self):
        ob = self.get_ob()
        return np.amax(ob)