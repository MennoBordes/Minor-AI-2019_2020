import gym
import json
import numpy as np
import gym_xplane.parameters as params
import gym_xplane.envs.xpc as xpc
from time import sleep, clock
from gym import spaces
from scipy.spatial.distance import pdist

class XplaneENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=5000, test=False):
        self.value = "not yet implemented"
        self.action_space = spaces.Dict({"Latitudinal Stick": spaces.Box(low=-1, high=1, shape=()),
                                         "Longitudinal Stick": spaces.Box(low=-1, high=1, shape=()),
                                         "Rudder Pedals": spaces.Box(low=-1, high=1, shape=()),
                                         "Throttle": spaces.Box(low=-1, high=1, shape=()),
                                         "Flaps": spaces.Box(low=0, high=1, shape=()),
                                         "Gear": spaces.Discrete(2),
                                         "Speedbrakes": spaces.Box(low=-0.5, high=1.5, shape=())})
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
        self.max_episode_steps = max_episode_steps
        self.stateLength = 14
        self.actions = [0, 0, 0, 0]
        self.test = test
        self.waypoints = []
        # Setup and test connection

        try:
            self.client = xpc.XPlaneConnect()
        except:
            print("connection error, check if xplane is running")
            raise Exception("connection error, check if xplane is running")
        print("I am client: ", self.client)

        # Setup initial run
        self.position = self.client.getPOSI(0)
        self.start_position = self.position
        self.start_state = self.getState()

        # Increase simulation speed
        self.client.sendDREF('sim/time/sim_speed', 500)

    def getState(self):
        state =  self.client.getDREFs(self.parameters.stateVariable)
        return state

    def calcReward(self, target_position, current_position, sigma=0.45):
        '''
        input : target state (a list containing the target heading, altitude and runtime)
                xplane_state(a list containing the aircraft heading , altitude at present timestep, and the running time)
                Note: if the aircraft crashes then the run time is small, thus the running time captures crashes
        output: Gaussian kernel similarîty between the two inputs. A value between 0 and 1
        '''

        data = np.array([target_position, current_position])

        pairwise_dists = pdist(data, 'cosine')
        # print('pairwise distance',pairwise_dists)
        similarity = np.exp(-pairwise_dists ** 2 / sigma ** 2)

        return pairwise_dists


    def step(self, actions):
        self.parameters.flag = False

        reward = -1
        actions_ = []
        margin = [3.5, 15]


        j = 0
        with xpc.XPlaneConnect() as client:
            try:

                reward += self.calcReward()
            except:
                print("except")

    def reset(self):

        with xpc.XPlaneConnect() as client:

            # Repair aircraft
            client.sendDREF("sim/operation/fix_all_systems", 1)

            # Reset position of the player aircraft
            client.sendPOSI(self.start_position)

            # set controls
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
            # Re-apply landing gear switch
            client.sendDREF("sim/cockpit2/controls/gear_handle_down", 0)

            # print("Set camera")
            sleep(1)


    def _get_info(self):
        """Returns a dictionary containing debug info."""
        return {"Control Parameters": self.parameters, "Actions": self.action_space}

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