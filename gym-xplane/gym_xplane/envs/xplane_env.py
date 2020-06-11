import gym
from time import sleep
from gym import spaces
import gym_xplane.envs.xpc2 as xpc
import gym_xplane.space_definition as envSpaces
import gym_xplane.parameters as parameters
import numpy as np
import math
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
        # self.action_space = spaces.Box(np.array([-1, -1, -1, -1 / 4]), np.array([1, 1, 1, 1]))
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1, 0, 0, -0.5]), np.array([1, 1, 1, 1, 1, 1, 1.5]))
        # self.action_space = spaces.Dict({"Latitudinal_Stick": spaces.Box(low=-1, high=1, shape=()),
        #                                  "Longitudinal_Stick": spaces.Box(low=-1, high=1, shape=()),
        #                                  "Rudder_Pedals": spaces.Box(low=-1, high=1, shape=()),
        #                                  "Throttle": spaces.Box(low=-1, high=1, shape=()),
        #                                  "Gear": spaces.Discrete(2),
        #                                  "Flaps": spaces.Box(low=0, high=1, shape=()),
        #                                  "Speedbrakes": spaces.Box(low=-0.5, high=1.5, shape=())})
        # self.observation_space = spaces.Box(
        #                           np.array([-360, -360, 0, -290, -100, -360, -360, -1000, -1300, -1000, -1000]),
        #                           np.array([360, 360, 8500, 290, 100, 360, 360, 1000, 1300, 1000, 1000]))
        self.observation_space = spaces.Dict({"Latitude": spaces.Box(low=0, high=360, shape=()),
                                              "Longitude": spaces.Box(low=0, high=360, shape=()),
                                              "Altitude": spaces.Box(low=0, high=8500, shape=()),
                                              "Pitch": spaces.Box(low=-290, high=290, shape=()),
                                              "Roll": spaces.Box(low=-100, high=100, shape=()),
                                              "Heading": spaces.Box(low=0, high=360, shape=()),
                                              "gear": spaces.Discrete(2),
                                              "speed": spaces.Box(low=-2205, high=2205, shape=()),
                                              "waypoint_lat": spaces.Box(low=0, high=360, shape=()),
                                              "waypoint_lon": spaces.Box(low=0, high=360, shape=()),
                                              "waypoint_alt": spaces.Box(low=0, high=8500, shape=())})
        # "yoke_pitch_ratio": spaces.Box(low=-2.5, high=2.5, shape=()),
        # "yoke_roll_ratio": spaces.Box(low=-300, high=300, shape=()),
        # "yoke_heading_ratio": spaces.Box(low=-180, high=180, shape=()),
        # "alpha": spaces.Box(low=-100, high=100, shape=()),
        # "wing_sweep_ratio": spaces.Box(low=-100, high=100, shape=()),
        # "flap_ratio": spaces.Box(low=-100, high=100, shape=()))
        # "speed": spaces.Box(low=-2205, high=2205, shape=())})

        self.ControlParameters.episodeStep = 0
        self.max_episode_steps = max_episode_steps
        self.stateLength = 10
        self.actions = [0, 0, 0, 0]
        self.test = test
        self.waypoints = []
        self.last_waypoints = []
        self.waypoint_goal = 1
        try:
            XplaneENV.CLIENT = Initial.connect(clientAddr, xpHost, xpPort, clientPort, timeout, max_episode_steps)
        except:
            print("connection error, check if xplane is running")
            raise Exception("connection error, check if xplane is running")
        print("I am client: ", XplaneENV.CLIENT)

        # Increase simulation speed
        XplaneENV.CLIENT.sendDREF('sim/time/sim_speed', 500)
        # Get starting position
        self.start_position = XplaneENV.CLIENT.getPOSI()
        self.ControlParameters.stateAircraftPosition = list(XplaneENV.CLIENT.getPOSI())
        self.previous_position = self.start_position

    def close(self):
        XplaneENV.CLIENT.close()

    def step(self, actions, AIType):
        self.ControlParameters.flag = False  # For synchronization during training

        reward = 0

        try:
            XplaneENV.CLIENT.sendCTRL(actions)  # send action
            sleep(0.0005)
            self.actions = actions

            state = []

            self.ControlParameters.stateAircraftPosition = list(XplaneENV.CLIENT.getPOSI())
            # print("planePosition: {}".format(self.ControlParameters.stateAircraftPosition))
            state = self.ControlParameters.stateAircraftPosition  # + self.waypoints[self.waypoint_goal]
            # print(state)
            # Add planeSpeed to state
            planeSpeed = XplaneENV.CLIENT.getDREF(self.ControlParameters.aircraftSpeed)
            state.append(planeSpeed[0])

            # Add target waypoint to state
            state.extend(self.waypoints[self.waypoint_goal].tolist())
            # print(state)

            # ******************************* Reward Parameters *********************************
            # Compare current position with previous position relative to the target
            if self.closer_lat(self.previous_position, self.ControlParameters.stateAircraftPosition,
                               self.waypoints[self.waypoint_goal]):
                reward += 0.2

            if self.closer_lon(self.previous_position, self.ControlParameters.stateAircraftPosition,
                               self.waypoints[self.waypoint_goal]):
                reward += 0.2

            if self.closer_alt(self.previous_position, self.ControlParameters.stateAircraftPosition,
                               self.waypoints[self.waypoint_goal]):
                reward += 0.1

            # Update previous position to current position
            self.previous_position = self.ControlParameters.stateAircraftPosition

            if AIType == AI_type.Cruise:
                # Check altitude
                if self.alt_anomaly(self.ControlParameters.stateAircraftPosition):
                    reward -= 0.1

                # Check if the plane has crashed
                if self.has_crashed():
                    reward -= 100
                    self.ControlParameters.flag = True

                # Check if the plane has reached the target waypoint
                if self.reached_waypoint(state):
                    reward += 10
                    self.waypoint_goal += 1

                # Check if the plane has reached the endpoint
                if self.reached_goal(state):
                    reward += 50
                    self.ControlParameters.flag = True
            return np.array(state), round(reward, 1), self.ControlParameters.flag, self._get_info()
        except Exception as e:
            print(f"ERROR: {e.__class__} \nText: {str(e)}")
            return np.array([]), round(reward, 1), False, self._get_info()

    def alt_anomaly(self, current_pos):
        """Checks if the current altitude is below a target"""
        current_altitude = current_pos[2]
        if current_altitude < 500 or current_altitude > 14_000:
            return True
        return False

    def closer_lat(self, previous_pos, current_pos, target_waypoint):
        # Get closest to target
        temp_list = [previous_pos[0], current_pos[0]]
        # Get the index  of the value closer to the target
        index = temp_list.index(min(temp_list, key=lambda x: abs(x - target_waypoint[0])))
        # If index = 1 then current position is closer than previous position
        if index == 1:
            return True
        return False

    def closer_lon(self, previous_pos, current_pos, target_waypoint):
        # Get closest to target
        temp_list = [previous_pos[1], current_pos[1]]
        # Get the index  of the value closer to the target
        index = temp_list.index(min(temp_list, key=lambda x: abs(x - target_waypoint[1])))
        # If index = 1 then current position is closer than previous position
        if index == 1:
            return True
        return False

    def closer_alt(self, previous_pos, current_pos, target_waypoint):
        # Get closest to target
        temp_list = [previous_pos[2], current_pos[2]]
        # Get the index  of the value closer to the target
        index = temp_list.index(min(temp_list, key=lambda x: abs(x - target_waypoint[2])))
        # If index = 1 then current position is closer than previous position
        if index == 1:
            return True
        return False

    def has_crashed(self):
        # Check if wheels are blown
        wheel_value = []
        for i in self.ControlParameters.wheelFailures:
            value = XplaneENV.CLIENT.getDREF(i)
            wheel_value.append(value)

        # Check if any wheelValue > 1 which would indicate a blown tire
        wheel_crash = (any(x[0] > 1.0 for x in wheel_value))

        if wheel_crash:
            return True

        # Check if the plane is on the ground
        on_ground = round(XplaneENV.CLIENT.getDREF("sim/flightmodel/failures/onground_any")[0]) > 0

        if on_ground:
            return True

        # Possibility for other checks

        return False

    def reached_waypoint(self, state):
        """Checks if the plane has reached the target waypoint"""
        # abs_tot = the allowed difference between
        close_latitude = math.isclose(state[0], self.waypoints[self.waypoint_goal][0], abs_tol=0.0009)
        close_longitude = math.isclose(state[1], self.waypoints[self.waypoint_goal][1], abs_tol=0.0009)
        close_altitude = math.isclose(state[2], self.waypoints[self.waypoint_goal][2], abs_tol=10)

        if close_latitude & close_altitude & close_longitude:
            return True

        return False

    def reached_goal(self, state):
        """Checks if the plane has reached the last waypoint"""
        close_latitude = math.isclose(state[0], self.last_waypoints[0], abs_tol=0.0009)
        close_longitude = math.isclose(state[1], self.last_waypoints[1], abs_tol=0.0009)
        close_altitude = math.isclose(state[2], self.last_waypoints[2], abs_tol=10)

        if close_latitude & close_altitude & close_longitude:
            return True

        return False

    def reset(self):
        """
        Reset environment and prepare for new episode
        :return: Initial state of reset environment
        """

        # Reset xplane env
        XplaneENV.CLIENT.resetPlane()

        reset_finished = False
        plane_position = []
        # Wait until finished loading
        while not reset_finished:
            try:
                # Try retrieving multiple datarefs
                XplaneENV.CLIENT.getDREF("sim/test/test_float")
                plane_position = XplaneENV.CLIENT.getPOSI()
                reset_finished = True
            except:
                print('Resetting environment...')

        XplaneENV.CLIENT.pauseSim(True)
        sleep(2)

        # XplaneENV.CLIENT.pauseSim(False)
        # plane_position = XplaneENV.CLIENT.getPOSI()
        self.start_position = plane_position

        # Reset time to 10:00 (32400.0)
        XplaneENV.CLIENT.sendDREF("sim/time/zulu_time/sec", 32400.0)
        # XplaneENV.CLIENT.sendDREF("sim/cockpit/switches/gear_handle_status", 1)

        # Reset variables / parameters
        self.ControlParameters.stateAircraftPosition = list(plane_position)
        self.waypoint_goal = 1
        self.actions = [0, 0, 0, 0]

        # Get plane position for state
        state = list(plane_position)

        # Add planeSpeed to state
        planeSpeed = XplaneENV.CLIENT.getDREF(self.ControlParameters.aircraftSpeed)
        state.append(planeSpeed[0])

        # Add target waypoint to state
        state.extend(self.waypoints[self.waypoint_goal].tolist())

        XplaneENV.CLIENT.pauseSim(False)
        # return state
        return np.array(state)

    def _get_info(self):
        """Returns a dictionary containing debug info."""
        return {"Control Parameters": self.ControlParameters, "Actions": self.action_space}

    def render(self, mode='human', close=False):
        pass

    def add_waypoints(self, json_path, land_start=True):
        waypoints = []

        with open(json_path) as json_file:
            nodes = json.load(json_file)
            data = nodes['nodes']
            for index, data in enumerate(data):
                if index is 0:
                    # Set first waypoint to starting position
                    waypoints.append(self.start_position[0])
                    waypoints.append(self.start_position[1])
                    waypoints.append(self.start_position[2])
                    if land_start:
                        # Add waypoints for Schiphol end of runway 18R
                        waypoints.append(52.3286247253418)  # Latitude
                        waypoints.append(4.708907604217529)  # Longitude
                        waypoints.append(150)  # Altitude
                    continue
                # Add waypoints from file
                waypoints.append(data['lat'])
                waypoints.append(data['lon'])
                # Calculate altitude
                alt = data['alt']
                if land_start:
                    alt = round(alt / 3.2808)
                waypoints.append(alt)

        XplaneENV.CLIENT.sendWYPT(op=1, points=waypoints)

        # Change waypoints to 3d array (n,3)
        self.waypoints = np.asarray(waypoints, dtype=float).reshape(round((len(waypoints)) / 3), 3)
        self.last_waypoints = self.waypoints[-1]

    def remove_waypoints(self):
        XplaneENV.CLIENT.sendWYPT(op=3, points=[])


class AI_type(object):
    TakeOff = 0
    Cruise = 1
    Landing = 2
