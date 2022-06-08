from gym.envs.registration import register

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=100,
    entry_point='deep_sprl.environments.contextual_point_mass_2d:ContextualPointMass2D'
)

register(
    id='Maze-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.maze:MazeEnv'
)
