from gym.envs.registration import register


multiplier = 1000
register(
    id='CartPole-long-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': multiplier * 200},
    reward_threshold= multiplier * 195.0,
)