#!/usr/bin/env python
# -*- coding: utf-8 -*-

from absl import flags
import gym
from src.lib.tester import run_agent

flags.DEFINE_string("agent", "q_network", "Agent do wywołania")

flags.DEFINE_string("env", "CartPole-v0", "Środowisko openai")


import sys
flags.FLAGS(sys.argv)

def get_class_name(module_name):
   words = module_name.split('_')
   return ''.join(word.capitalize() for word in words)

import importlib
agent_name = flags.FLAGS.agent + "_agent"
env_name = flags.FLAGS.env

agent_module = importlib.import_module("src.agents." + env_name + "." + agent_name )
agent_class = getattr(agent_module, get_class_name(agent_name))

print("Agent: " + flags.FLAGS.agent)

agent = agent_class()

run_agent(agent, gym.make(env_name))