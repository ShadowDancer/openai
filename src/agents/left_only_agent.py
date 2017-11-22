from src.lib.base_agent import BaseAgent

class left_only_agent(BaseAgent):
    """Test agent moving left"""

    x = 0
    def action(self, observation, action_space):
        self.x = 1-self.x
        return self.x

    def learn(sefl, observation, reward, action):
        pass