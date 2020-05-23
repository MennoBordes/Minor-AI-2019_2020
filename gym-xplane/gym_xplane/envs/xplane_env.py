import gym
from time import sleep, clock

import gym_xplane.envs.xpc2 as xpc
import gym_xplane.space_definition as envSpaces
import gym_xplane.parameters as parameters
import numpy as np
import json


class Initial:
    def connect(clientAddr, xpHost, xpPort, clientPort, timeout, max_episode_steps):
        return xpc.XPlaneConnect(clientAddr, xpHost, xpPort, clientPort, timeout, max_episode_steps)


class XplaneENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, clientAddr, xpHost, xpPort, clientPort, timeout=3000, max_episode_steps=5000, test=False):
        XplaneENV.CLIENT = None

        EnvSpace = envSpaces.xplane_space()

        self.ControlParameters = parameters.getParameters()
        self.action_space = EnvSpace.Action_space()
        self.observation_space = EnvSpace.Observation_space()

        self.ControlParameters.episodeStep = 0
        self.max_episode_steps = max_episode_steps
        self.stateLength = 10
        self.actions = [0, 0, 0, 0]
        self.test = test
        self.waypoints = []
        try:
            XplaneENV.CLIENT = Initial.connect(clientAddr, xpHost, xpPort, clientPort, timeout, max_episode_steps)
        except:
            print("connection error, check if xplane is running")
            raise Exception("connection error, check if xplane is running")
        print("I am client: ", XplaneENV.CLIENT)
        # Increase simulation speed
        XplaneENV.CLIENT.sendDREF('sim/time/sim_speed', 500)
        self.position = XplaneENV.CLIENT.getPOSI()

    def close(self):
        XplaneENV.CLIENT.close()

    def step(self, actions):
        self.ControlParameters.flag = False  # For synchronization during training

        reward = -1
        actions_ = []
        margin = [3.5, 15]  # Margin allowed on altitude and heading

        j = 0  # Getting simulation timing measurement

        try:
            # print("prevous action", self.actions)  # prvious ation
            # print("action on ctrl ...", XplaneENV.CLIENT.getCTRL())  # action on control surface
            i = clock()

            XplaneENV.CLIENT.sendCTRL(actions)  # send action
            sleep(0.0005)
            self.actions = actions

            state = []
            state2 = []

            stateVariableTemp = XplaneENV.CLIENT.getDREFs(self.ControlParameters.stateVariable)
            self.ControlParameters.stateAircraftPosition = list(XplaneENV.CLIENT.getPOSI())
            self.ControlParameters.stateVariableValue = [i[0] for i in stateVariableTemp]

            state = self.ControlParameters.stateAircraftPosition + self.ControlParameters.stateVariableValue
            # print("state 5", state[5])

            # ******************************* Reward Parameters *********************************
            rewardVector = XplaneENV.CLIENT.getDREF(self.ControlParameters.rewardVariable)[0]
            # print(rewardVector)
            # ***********************************************************************************

            P = XplaneENV.CLIENT.getDREF("sim/flightmodel/position/P")[0]
            Q = XplaneENV.CLIENT.getDREF("sim/flightmodel/position/Q")[0]
            R = XplaneENV.CLIENT.getDREF("sim/flightmodel/position/R")[0]
            print("P", P, "Q", Q, "R", R)

            return np.array(state2), reward, self.ControlParameters.flag, self._get_info()
        except:
            print("except")

        print("STEP not yet implemented")

    def reset(self):
        """
        Reset environment and prepare for new episode
        :return: Initial state of reset environment
        """
        # Reset xplane env
        XplaneENV.CLIENT.resetPlane()
        reset_finished = False
        # Wait until finished loading
        while not reset_finished:
            try:
                XplaneENV.CLIENT.getDREF("sim/test/test_float")
                reset_finished = True
            except:
                print('Resetting environment.')
                pass
        sleep(2)
        # Reset time to 10:00 (32400.0)
        XplaneENV.CLIENT.sendDREF("sim/time/zulu_time/sec", 32400.0)
        # XplaneENV.CLIENT.sendDREF("sim/cockpit/switches/gear_handle_status", 1)

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

    def _get_info(self):
        """Returns a dictionary containing debug info."""
        return {"Control Parameters": self.ControlParameters, "Actions": self.action_space}

    def render(self, mode='human', close=False):
        pass

    def add_waypoints(self, json_path):
        waypoints = []

        with open(json_path) as json_file:
            nodes = json.load(json_file)
            data = nodes['nodes']
            for index, data in enumerate(data):
                if index is 0:
                    # Set first waypoint to starting position
                    waypoints.append(self.position[0])
                    waypoints.append(self.position[1])
                    waypoints.append(self.position[2])

                    # Add waypoints for Schiphol end of runway 18R
                    waypoints.append(52.3286247253418)      # Latitude
                    waypoints.append(4.708907604217529)     # Longitude
                    waypoints.append(150)                   # Altitude
                    continue
                # Add waypoints from file
                waypoints.append(data['lat'])
                waypoints.append(data['lon'])
                waypoints.append((data['alt'] / 3.2808))

        self.waypoints = waypoints
        XplaneENV.CLIENT.sendWYPT(op=1, points=waypoints)

    def remove_waypoints(self):
        XplaneENV.CLIENT.sendWYPT(op=3, points=[])
        pass
