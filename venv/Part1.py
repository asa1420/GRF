import gfootball.env as football_env
env = football_env.create_environment(env_name="academy_empty_goal", representation="pixels", render=True)
state = env.reset()
state_dims = env.observation_space.shape
print(state_dims)
n_actions = env.action_space.n
print(n_actions)