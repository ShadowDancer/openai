from src.lib.base_agent import BaseAgent

class LeftOnlyAgent(BaseAgent):
    """Test agent moving left"""

    x = 0
    def act(self, observation, action_space):
        self.x = 1-self.x
        return self.x