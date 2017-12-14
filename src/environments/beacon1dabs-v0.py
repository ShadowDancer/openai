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

class Beacon1DAbsolute(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def _observation(self):


        obs = np.zeros(self.size)
        obs[self.beacon] = 1
        obs[self.player] = 2
        return obs

    def __init__(self):
        self.size = 64
        self.beacon = 0
        self.player = 0

        self.action_space = spaces.Discrete(self.size)
        self.observation_space = spaces.Discrete(self.size)

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
            if action < self.player and self.player > 0:
                self.player = self.player - 1
            if action > self.player and self.player < self.size - 1:
                self.player = self.player + 1



        done = (self.player == self.beacon) or (self.reward == 0)

        if not done:
            if self.reward > 0:
                self.reward = self.reward - 1

        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1

        obs = self._observation()

        if self.viewer:
            print('P', str(self.player).zfill(2), ' T', str(self.beacon).zfill(2), ' R', str(self.reward).zfill(2), ' A', action, ' ', done)

        return obs, self.reward, done, {}

    def _reset(self):
        self.reward = 64
        self.beacon = int(self.np_random.uniform(0, 1, 1)[0] * self.size)
        self.player = int((self.beacon + self.size / 2) % self.size)
        self.steps_beyond_done = None
        return self._observation()

    def _render(self, mode='human', close=False):
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
            
            for i in range(self.size):
                l,r,t,b = -dotsize/2 + offsetx, dotsize/2 + offsetx, dotsize/2 + offsety, -dotsize/2 + offsety
                poly = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                #polyTrans = rendering.Transform()
                #polyTrans.set_translation(i * dotsize, dotsize/2.0)
                #poly.add_attr(polyTrans)
                polygons.append(poly)
                self.viewer.add_geom(poly)

        for i in range(self.size):
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
    id='beacon1dabs-v0',
    entry_point=Beacon1DAbsolute,
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    reward_threshold=5 * 195.0,
)