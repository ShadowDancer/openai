#!/usr/bin/env python
# -*- coding: utf-8 -*-

from absl import flags
import gym
from gym import wrappers
from src.lib.tester import run_agent
import importlib
from src.environments import *


flags.DEFINE_string("agent", "CartPole-v0/q_network", "Agent do wywołania")

flags.DEFINE_string("env", "CartPole-v0", "Środowisko openai")


import sys
flags.FLAGS(sys.argv)

def get_agent(env_name):
    def get_class_name(module_name):
        words = module_name.split('_')
        return ''.join(word.capitalize() for word in words)

    split = flags.FLAGS.agent.split('/')
    agent_name =  split[1] + "_agent"

    env_split = split[0].split('-')
    if len(env_split) >  1:
        env_split[0] = env_split[0] + '-' + env_split[-1]

    agent_module = importlib.import_module("src.agents." + env_split[0] + "." + agent_name )
    agent_class = getattr(agent_module, get_class_name(agent_name))
    return agent_class()

print("Agent: " + flags.FLAGS.agent)

def run(learn=True):
    env_name = flags.FLAGS.env
    with get_agent(env_name) as agent:
        #scr_env =  gym.make(env_name)
        #id = scr_env.env._spec
        #env = wrappers.Monitor(scr_env, '/tmp/cartpole-experiment-1', force=True)
        #env.env._spec = id
        env =  gym.make(env_name)
        run_agent(agent, env, learn=learn)



