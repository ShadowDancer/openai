from src.lib.base_agent import BaseAgent


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt


class deepq_test_agent(BaseAgent):
    """Test agent executing random ctions"""
    def action(self, observation, action_space):
        return action_space.sample()

    def learn(sefl, observation, reward, action):
        pass