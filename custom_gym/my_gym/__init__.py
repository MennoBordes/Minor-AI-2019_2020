from gym.envs.registration import register

register(
    id="xplane-v0",
    entry_point="my_gym.envs:xplane"
        )