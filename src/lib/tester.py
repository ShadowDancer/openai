import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "CartPole-v0", "Środowisko openai")
flags.DEFINE_integer("episodes", 2000, "Ilość gier do zagrania")
flags.DEFINE_integer("frames", 4000, "Maksymalna ilość klatek/gra")
flags.DEFINE_float("visualize", 0, "Czy renderować grę")

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def run_agent(agent, learn=True):
    """Executes predefined number of steps, actions are choosen by agent"""
    """If learn = true learn method will be called on agent """
    env = gym.make(FLAGS.env)
    env.reset()
    rewards = []

    agent.setup()
    # Each of these is its own game.
    try:
        for episode in range(FLAGS.episodes):
            observation = env.reset()
            # this is each frame, up to 200...but we wont make it that far.
            for t in range(FLAGS.frames):
                # This will display the environment
                # Only display if you really want to see it.
                # Takes much longer to display it.
                if FLAGS.visualize != 0:
                    env.render()

                action = agent.action(observation, env.action_space)
                observation, reward, done, info = env.step(action)
                if done:
                    rewards.append(reward)
                    if len(rewards) > 100:
                        rewards.pop(0)
                    if episode > 0 and ((episode + 1) % 100 == 0):
                        print("100 episodes mean reward: " + str(mean(rewards)))
                    break
                if learn:
                    agent.learn(observation, reward, action)
    finally:
        agent.dispose()
