import gym
from time import sleep
import numpy as np

import gym_xplane.envs.xpc as xpc
from gym import error, spaces, utils
from gym.utils import seeding


class XplaneENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.value = "not yet implemented"

    def step(self, action):
        print("Not yet implemented")

    def reset(self):
        """
        Reset environment and prepare for new episode
        :return: Initial state of reset environment
        """
        with xpc.XPlaneConnect() as client:
            # client.sendDREF("sim/cockpit/switches/gear_handle_status", 1)
            # Reset position of the player aircraft
            #       Lat               Lon                Alt            Pitch Roll Yaw Gear
            posi = [52.3286247253418, 4.708916664123535, -0.315114825963974, 0, 0, 0, 1]
            client.sendPOSI(posi)

            # print("setting controls")
            ctrl = [0.0, 0.0, 0.0, 0.0, 1, 0]
            client.sendCTRL(ctrl)

            '''
            Reset velocity (3)
            Reset heading (17)
            '''
            data = [ \
                [3, 0, 0, 0, 0, -998, -998, -998, -998], \
                [17, 0, 0, 3.1591200828552246, 0, -998, -998, -998, -998], \
            ]
            client.sendDATA(data)
            # Reset time to 10:00 (32400.0)
            client.sendDREF("sim/time/zulu_time_sec", 32400.0)
            # Re-apply parking brake
            client.sendDREF("sim/cockpit2/controls/parking_brake_ratio", 1)

            # print("Set camera")
            client.sendVIEW(xpc.ViewType.Chase)
            sleep(0.5)

        print("No RETURN implemented yet")

    def render(self, mode='human', close=False):
        print("Not yet implemented")
