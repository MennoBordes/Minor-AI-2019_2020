import gym
import json
import numpy as np
import gym_xplane.parameters as params
import gym_xplane.envs.xpc as xpc
from time import sleep, clock
from gym import error, spaces, utils
from gym.utils import seeding

class Initial:
    def connect(self, client_addr, xp_host, xp_port, client_port, timeout, max_steps):
        return xpc.XPlaneConnect(client_addr, xp_host, xp_port, client_port, timeout, max_steps)

class XplaneENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.value = "not yet implemented"
        self.action_space = spaces.Dict({"Latitudinal Stick": spaces.Box(low=-1, high=1, shape=()),
                                         "Longitudinal Stick": spaces.Box(low=-1, high=1, shape=()),
                                         "Rudder Pedals": spaces.Box(low=-1, high=1, shape=()),
                                         "Throttle": spaces.Box(low=-1, high=1, shape=()),
                                         "Flaps": spaces.Box(low=0, high=1, shape=()),
                                         "Gear": spaces.Discrete(2)})
        self.observation_space = spaces.Dict({"Latitude":  spaces.Box(low=0, high=360, shape=()),
                                              "Longitude":  spaces.Box(low=0, high=360, shape=()),
                                              "Altitude":  spaces.Box(low=0, high=8500, shape=()),
                                              "Pitch":  spaces.Box(low=-290, high=290, shape=()),
                                              "Roll":  spaces.Box(low=-100, high=100, shape=()),
                                              "Heading":  spaces.Box(low=0, high=360, shape=()),
                                              "gear":  spaces.Discrete(2),
                                              "yoke_pitch_ratio":  spaces.Box(low=-2.5, high=2.5, shape=()),
                                              "yoke_roll_ratio":  spaces.Box(low=-300, high=300, shape=()),
                                              "yoke_heading_ratio":  spaces.Box(low=-180, high=180,shape=()),
                                              "alpha":  spaces.Box(low=-100, high=100,shape=()),
                                              "wing_sweep_ratio":  spaces.Box(low=-100, high=100, shape=()),
                                              "flap_ratio":  spaces.Box(low=-100, high=100, shape=()),
                                              "speed": spaces.Box(low=-2205, high=2205, shape=())})
        self.parameters = params.getParameters()
        self.waypoints = []
        try:
            self.client = xpc.XPlaneConnect(xpHost="192.168.0.1", xpPort=49000)
        except:
            print("connection error, check if xplane is running")
            raise Exception("connection error, check if xplane is running")
        print("I am client: ", self.client)
        # Increase simulation speed
        self.client.sendDREF('sim/time/sim_speed', 500)
        self.position = self.client.getPOSI()

    def step(self, actions):
        self.parameters.flag = False

        reward = -1
        actions_ = []
        margin = [3.5, 15]

        j = 0
        with xpc.XPlaneConnect() as client:
            try:
                i = clock()
                self.actions = actions

                state1 = []
                state2 = []

                stateVariableTemp =client.getDREFs(self.parameters.stateVariable)
                self.parameters.stateAircraftPosition = list(self.client.getPOSI())
                self.parameters.stateVariableValue = [i[0] for i in stateVariableTemp]

                state1 = self.parameters.stateAircraftPosition + self.parameters.stateVariableValue

                # ******************************* Reward Parameters *********************************
                rewardVector = client.getDREF(self.parameters.rewardVariable)[0]
                # print(rewardVector)
                # ***********************************************************************************

                P = client.getDREF("sim/flightmodel/position/P")[0]
                Q = client.getDREF("sim/flightmodel/position/Q")[0]
                R = client.getDREF("sim/flightmodel/position/R")[0]
                print("P", P, "Q", Q, "R", R)

                return np.array(state2), reward, self.parameters.flag, self._get_info()
            except:
                print("except")

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

    def _get_info(self):
        """Returns a dictionary containing debug info."""
        return {"Control Parameters": self.parameters, "Actions": self.action_space}

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
                    waypoints.append(52.3286247253418)  # Latitude
                    waypoints.append(4.708907604217529)  # Longitude
                    waypoints.append(150)  # Altitude
                    continue
                # Add waypoints from file
                waypoints.append(data['lat'])
                waypoints.append(data['lon'])
                waypoints.append((data['alt'] / 3.2808))

        self.waypoints = waypoints
        self.client.sendWYPT(op=1, points=waypoints)

    def remove_waypoints(self):
        self.client.sendWYPT(op=3, points=[])
        pass