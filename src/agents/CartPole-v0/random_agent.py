from src.lib.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Test agent executing random ctions"""
    def act(self, observation, action_space):
        return action_space.sample()