import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
from absl import flags
import time

FLAGS = flags.FLAGS
flags.DEFINE_integer("episodes", 100 * 100, "Ilość gier do zagrania")
flags.DEFINE_integer("frames", 4000, "Maksymalna ilość klatek/gra")
flags.DEFINE_float("visualize", 0, "Czy renderować grę")

def format_float(f):
    return format(f, '07.2f')

def run_agent(agent, env):
    """Executes predefined number of steps, actions are choosen by agent, environment should be openai gym environemnt"""
    env.reset()
    rewards = []
    means = []
    start = time.time()
    

    agent.setup()
    # Each of these is its own game.
    try:
        for episode in range(FLAGS.episodes):
            observation = env.reset()
            # this is each frame, up to 200...but we wont make it that far.
            current_reward = 0
            for t in range(FLAGS.frames):
                # This will display the environment
                # Only display if you really want to see it.
                # Takes much longer to display it.
                if FLAGS.visualize != 0:
                    env.render()

                action = agent.act(observation, env.action_space)
                observation, reward, done, info = env.step(action)
                agent.observe(observation, reward, action)
                current_reward += reward
                if done:
                    rewards.append(current_reward)
                    if len(rewards) > 100:
                        rewards.pop(0)
                    if episode > 0 and ((episode + 1) % 100 == 0):
                        mean = np.mean(rewards)
                        means.append(mean)
                        print("100 episodes reward mean: " + format_float(mean) )
                    break
            agent.next_episode(episode)
            
        end = time.time()
        print("Finished with best mean: " + format_float(max(means)) + " in " + str(end-start) + "s")
    finally:
        agent.dispose()
