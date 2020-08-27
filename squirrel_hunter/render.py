import pygame
import SETTINGS
import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras import models
import numpy as np
from skimage.transform import resize

class Render:

    def __init__(self, surface, map_surface):
        self.fps = ''
        self.sc = surface
        self.sc_map = map_surface

        self.texture_kit = {
            '1': [pygame.image.load(
                f'img/textures/128px/var_9/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            '2': [pygame.image.load(
                f'img/textures/128px/var_2/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            '3': [pygame.image.load(
                f'img/textures/128px/var_3/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            '4': [pygame.image.load(
                f'img/textures/128px/var_4/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            '5': [pygame.image.load(
                f'img/textures/128px/var_12/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            '6': [pygame.image.load(
                f'img/textures/128px/var_10/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            '7': [pygame.image.load(
                f'img/textures/128px/var_11/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            '8': [pygame.image.load(
                f'img/textures/128px/var_1/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            'P': [pygame.image.load(
                f'img/textures/128px/var_p/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            'F': pygame.image.load(
                f'img/textures/128px/floor/5.jpg').convert(),
            'C': pygame.image.load(
                f'img/textures/128px/floor/4.jpg').convert(),
            'S': [pygame.image.load(
                f'img/textures/128px/sky/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
            'N': [pygame.image.load(
                f'img/textures/128px/var_n/{i}.png').convert() for i in range(1, SETTINGS.tile_size + 1)],
        }

        self.model = models.load_model('./seg_512_128.h5', compile=False)

    def draw_world(self, walls, floor):

        for n_ray in walls:
            '''Drawin floor'''
            if floor[n_ray]:
                for coords, area in floor[n_ray]:
                    self.sc.blit(self.texture_kit['F'], coords, area)
                    # self.sc.blit(self.texture_kit['C'], (coords[0], SETTINGS.HEIGHT - 12 - coords[1]), area)

            '''Drawing wall'''
            # _, projected_height, texture_offset, texture = walls[n_ray]
            projected_height = walls[n_ray][1]
            texture_offset = walls[n_ray][2]
            texture = walls[n_ray][3]
            img = pygame.transform.scale(self.texture_kit[texture][texture_offset],
                                         (SETTINGS.step_screen, projected_height))
            self.sc.blit(img, (SETTINGS.table_scale_screen[n_ray], SETTINGS.h_HEIGHT - projected_height // 2))

    def draw_map(self, ob):
        ob = np.transpose(ob[0])
        #array = np.zeros(shape=(128, 128, 3)) + 125
        array = np.random.randint(low=0, high=1, size=(128, 128, 3))
        array[:, :, 0] = np.where(ob == 9, 250, 0)
        array[:, :, 1] = np.where(ob == 3, 250, 0)
        array[:, :, 2] = np.where(ob == 2, 250, 0)
        pygame.pixelcopy.array_to_surface(self.sc_map, array)
        self.sc.blit(self.sc_map, SETTINGS.map_position)

    def draw_background(self, angle, x, y):
        # pygame.draw.rect(self.sc, SETTINGS.BLUE_SKY, (0, 0, SETTINGS.WIDTH, SETTINGS.h_HEIGHT))
        # pygame.draw.rect(self.sc, SETTINGS.DGRAY, (0, SETTINGS.h_HEIGHT, SETTINGS.WIDTH, SETTINGS.h_HEIGHT))
        [self.sc.blit(self.texture_kit['S'][int((math.degrees(angle) + i) % SETTINGS.tile_size)],
                      (i * 10, 0)) for i in range(SETTINGS.tile_size)]

    def display_fps(self, clock):
        font = pygame.font.SysFont("Arial", 28)
        #self.fps = clock.get_fps()
        #render = font.render(str(int(self.fps)), 0, SETTINGS.RED)
        render = font.render('reward = ' + str(clock), -10, SETTINGS.RED)
        self.sc.blit(render, SETTINGS.fps_coords)

    def mapping(self, coord):
        return (coord // SETTINGS.tile_size) * SETTINGS.tile_size

    def scaling_to_map(self, coord):
        return int(coord * SETTINGS.scale_map_player)

    def get_ob(self, sc):
        array = pygame.surfarray.array3d(sc)
        array = np.transpose(array, axes=[1, 0, 2])
        array = resize(array, (512, 512, 3))
        predictions = self.model.predict(np.array([array]))
        ob = np.argmax(predictions, axis=3)
        return ob

    def get_reward(self, ob):
        if np.count_nonzero(ob == 3) > 100:
            reward = -1
        elif np.count_nonzero(ob == 9) > 100:
            reward = 1
        else:
            reward = 0
        return reward
