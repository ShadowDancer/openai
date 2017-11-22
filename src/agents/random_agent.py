from src.lib.base_agent import BaseAgent

class random_agent(BaseAgent):
    """Test agent executing random ctions"""
    def action(self, observation, action_space):
        return action_space.sample()

    def learn(sefl, observation, reward, action):
        pass