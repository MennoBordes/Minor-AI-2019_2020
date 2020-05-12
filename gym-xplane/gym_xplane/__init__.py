from gym.envs.registration import register

register(
    id='xplane-gym-v0',
    entry_point='gym_xplane.envs:xplaneENV',
)
