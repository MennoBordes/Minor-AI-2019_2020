import gym
from time import sleep
import numpy as np

import gym_xplane.envs.xpc2 as xpc
import gym_xplane.space_definition as envSpaces
import gym_xplane.parameters as parameters
from gym import error, spaces, utils
from gym.utils import seeding


class Initial:
    def connect(clientAddr, xpHost, xpPort, clientPort, timeout, max_episode_steps):
        return xpc.XPlaneConnect(clientAddr, xpHost, xpPort, clientPort, timeout, max_episode_steps)


class XplaneENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, clientAddr, xpHost, xpPort, clientPort, timeout=3000, max_episode_steps=5000, test=False):
        XplaneENV.CLIENT = None

        _envspace = envSpaces.xplane_space()

        self.ControlParameters = parameters.getParameters()
        self.action_space = _envspace._action_space()
        self.observation_space = _envspace._observation_space()

        self.ControlParameters.episodeStep = 0
        self.max_episode_steps = max_episode_steps
        self.stateLength = 10
        self.action = [0, 0, 0, 0]
        self.test = test
        try:
            XplaneENV.CLIENT = Initial.connect(clientAddr, xpHost, xpPort, clientPort, timeout, max_episode_steps)

        except:
            print("connection error, check if xplane is running")
        print("I am client: ", XplaneENV.CLIENT)

        self.value = "not yet implemented"

    def close(self):
        XplaneENV.CLIENT.close()

    def step(self, action):
        print("Not yet implemented")

    def reset(self):
        """
        Reset environment and prepare for new episode
        :return: Initial state of reset environment
        """
        # Reset xplane env
        XplaneENV.CLIENT.resetPlane()
        # Reset time to 10:00 (32400.0)
        XplaneENV.CLIENT.sendDREF("sim/time/zulu_time/sec", 32400.0)
        # XplaneENV.CLIENT.sendDREF("sim/cockpit/switches/gear_handle_status", 1)
        print("gearhandle: ", XplaneENV.CLIENT.getDREF("sim/cockpit/switches/gear_handle_status"))

        ctrl = XplaneENV.CLIENT.getCTRL()
        print("control: ", ctrl)

        # Legacy for xpc.py
        if False:
            with xpc.XPlaneConnect() as client:

                client.resetPlane()

                # Reset time to 10:00 (32400.0)
                client.sendDREF("sim/time/zulu_time_sec", 32400.0)

                return
                # client.sendDREF("sim/operation/fix_all_systems", 1)
                client.sendDREF("sim/cockpit/switches/gear_handle_status", 1)
                client.sendDREF("sim/cockpit2/controls/gear_handle_down", 1)

                # Reset position of the player aircraft
                #       Lat               Lon                Alt            Pitch Roll Yaw Gear
                posi = [52.3286247253418, 4.708916664123535, -0.315114825963974, 0, 0, 0, 1]
                client.sendPOSI(posi)

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

    def _get_info(self):
        """Returns a dictionary containing debug info."""
        return {"Control Parameters": self.ControlParameters, "Actions": self.action_space}

    def render(self, mode='human', close=False):
        pass
