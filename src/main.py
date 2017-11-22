#!/usr/bin/env python
# -*- coding: utf-8 -*-

from absl import flags
import gym
from src.lib.tester import run_agent

flags.DEFINE_string("agent", "deepq_test", "Agent do wywołania")

import sys
flags.FLAGS(sys.argv)


import importlib
agent_name = flags.FLAGS.agent + "_agent"
agent_module = importlib.import_module("src.agents." + agent_name )
agent_class = getattr(agent_module, agent_name)

print("Agent: " + flags.FLAGS.agent)

agent = agent_class()

run_agent(agent)