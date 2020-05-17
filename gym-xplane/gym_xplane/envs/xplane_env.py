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

            client.resetPlane()

            return
            # client.sendDREF("sim/operation/fix_all_systems", 1)
            client.sendDREF("sim/cockpit/switches/gear_handle_status", 1)
            client.sendDREF("sim/cockpit2/controls/gear_handle_down", 1)

            # Reset position of the player aircraft
            #       Lat               Lon                Alt            Pitch Roll Yaw Gear
            posi = [52.3286247253418, 4.708916664123535, -0.315114825963974, 0, 0, 0, 1]
            client.sendPOSI(posi)

            # print("setting controls")
            ctrl = [0.0, 0.0, 0.0, 0.0, 1, 0]
            client.sendCTRL(ctrl)

            '''
            (3) Reset velocity
            (17) Reset heading
            (62) Reset fuel 
            '''
            data = [ \
                [3, 0, 0, 0, 0, -998, -998, -998, -998], \
                [17, 0, 0, 3.1591200828552246, 0, -998, -998, -998, -998], \
                [62, 37515, 14938, 42118, 42132, 14938, 4404.9, 4404.9, 11107], \
            ]
            client.sendDATA(data)
            # Reset time to 10:00 (32400.0)
            client.sendDREF("sim/time/zulu_time_sec", 32400.0)
            # Re-apply parking brake
            client.sendDREF("sim/cockpit2/controls/parking_brake_ratio", 1)
            # Re-apply landing gear switch
            client.sendDREF("sim/cockpit2/controls/gear_handle_down", 1)

            # print("Set camera")
            client.sendVIEW(xpc.ViewType.Chase)
            sleep(1)

        print("No RETURN implemented yet")

    def render(self, mode='human', close=False):
        pass
