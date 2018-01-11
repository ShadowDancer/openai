"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class Beacon2DRealtive(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    offsets = [
        (0, 1),
        (1, 1),
        (-1, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (0, -1),
        (1, -1)]


    def _flat_pos(self, pos, width):
        return pos[1] * width + pos[0]

    def _observation(self):
         obs = np.zeros(self.size)
         obs[self.beacon] = 1

         obs2 = np.zeros(self.size)
         obs2[self.player] = 1

         return np.concatenate([obs.flatten(), obs2.flatten()])

    def __init__(self):
        self.size = (5, 5)
        self.timer = 0
        self.beacon = (0, 0)
        self.player = (0, 0)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(0, 2, self.size)

        self._seed()
        self.viewer = None
        self.state = None
        self.reward = 100

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        if self.beacon != self.player:            
            offset = self.offsets[action]
            x = self.player[0] + offset[0]
            y = self.player[1] + offset[1]
            
            if x < 0:
                x = 0
                
            if y < 0:
                y = 0
                
            if x >= self.size[0]:
                x = self.size[0] - 1
                
            if y >= self.size[1]:
                y = self.size[1] - 1

            self.player = (x, y)

        done = (self.player == self.beacon) or (self.timer == 0)

        if not done:
            if self.timer > 0:
                self.timer = self.timer - 1
            self.reward = -1

        elif self.steps_beyond_done is None:
            if (self.player == self.beacon):
                self.reward = 1
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            self.reward = 0

        obs = self._observation()

       # print('P ', self._format_tuple(self.player), ' T ', self._format_tuple(self.beacon), ' R', str(self.reward).zfill(2), ' A', action, ':', self._format_tuple(self.offsets[action]), ' ', done)
        return obs, self.reward, done, {}

    def _format_tuple(self, t):
        return str(t[0]).zfill(2) + "," + str(t[1]).zfill(2)


    def _reset(self):
        self.timer = self.size[0] * 4
        self.reward = -1
        self.beacon = (int(self.np_random.uniform(0, 1, 1)[0] * self.size[0]), int(self.np_random.uniform(0, 1, 1)[0] * self.size[1]))
        self.player = (int(self.np_random.uniform(0, 1, 1)[0] * self.size[0]), int(self.np_random.uniform(0, 1, 1)[0] * self.size[1]))

        while self.player == self.beacon:
            self.player = (int(self.np_random.uniform(0, 1, 1)[0] * self.size[0]), int(self.np_random.uniform(0, 1, 1)[0] * self.size[1]))

        self.steps_beyond_done = None
        return self._observation()

    def _render(self, mode='human', close=False):
        return
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            polygons = []
            self.polygons = polygons


            dotsize = 20
            offsetx = 10
            offsety = 100
            
            for i in range(self.size[0]):
                l,r,t,b = -dotsize/2 + offsetx, dotsize/2 + offsetx, dotsize/2 + offsety, -dotsize/2 + offsety
                poly = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                #polyTrans = rendering.Transform()
                #polyTrans.set_translation(i * dotsize, dotsize/2.0)
                #poly.add_attr(polyTrans)
                polygons.append(poly)
                self.viewer.add_geom(poly)

        for i in range(self.size[0]):
            if self.beacon == i:
                self.polygons[i].set_color(.5,.5,.8)
            elif self.player == i:
                self.polygons[i].set_color(.8,.6,.4)
            else:
                self.polygons[i].set_color(0,0,0)

        if self.state is None: return None
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


from gym.envs.registration import register

register(
    id='beacon2drel-v0',
    entry_point=Beacon2DRealtive,
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    reward_threshold=5 * 195.0,
)