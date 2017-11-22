class BaseAgent:
    def action(self, observation, action_space):
        pass

    def learn(self, observation, reward, action):
        pass

    def setup(self):
        pass
    
    def dispose(self):
        pass