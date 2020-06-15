import gym
from gym import error, utils
from gym.utils import seeding
from custom_gym.envs.myxpc import xpc2 as xpc
from custom_gym.envs.myxpc.xplane_functions import get_waypoints
from gym.spaces import Box
import pygetwindow
import numpy as np
from pydirectinput import keyDown, keyUp
import time


class XPL(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        print("init xpl")

    def step(self, action):
        print("step")

    def reward(self):
        get_waypoints()

    def reset(self):
        print("reset")
        # Set simulation speed for faster training
        print("Setting up simulation")
        with xpc.XPlaneConnect() as client:
            # Verify connection
            try:
                # If X-Plane does not respond to the request, a timeout error
                # will be raised.
                client.getDREF("sim/test/test_float")
            except:
                print("Error establishing connection to X-Plane.")
                print("Exiting...")
                return
            simulation_dref = "sim/time/sim_speed"
            client.sendDREF(simulation_dref, 1000)
            res = client.getDREF(simulation_dref)
            print(res)
        # Selecting the current and XPlane window 
        current_window = pygetwindow.getActiveWindow()
        xplane_window = pygetwindow.getWindowsWithTitle("X-System")[0]
        # Focuss on the Xplane window
        xplane_window.activate()
        
        # Performing the reset command ctr+; on the focussed window
        keyDown('ctrl')
        keyDown(';')
        keyUp('ctrl')
        keyUp(';')
        # Return to the old window I was on
        current_window.activate()
    
    def state(self):
        print("Setting up simulation")
        with xpc.XPlaneConnect() as client:
            # Verify connection
            try:
                # If X-Plane does not respond to the request, a timeout error
                # will be raised.
                client.getDREF("sim/test/test_float")
            except:
                print("Error establishing connection to X-Plane.")
                print("Exiting...")
                return
            air_speed_dref = "sim/flightmodel/position/true_airspeed"
            air_speed = client.getDREF(air_speed_dref)
            position = np.array(client.getPOSI(0))
            observation = np.append(position, [air_speed])
        return observation
            

    def render(self, mode="human"):
        print("render")

    def quit(self):
        print("quit")

    def test_p(self):
        xplane_window = pygetwindow.getWindowsWithTitle("X-System")[0]
        xplane_window.activate()
        keyDown('p')
        keyUp('p')

    def actions(self):
        movements = []

    def get_waypoints(self):
        print("Setting up simulation")
        with xpc.XPlaneConnect() as client:
            # Verify connection
            try:
                # If X-Plane does not respond to the request, a timeout error
                # will be raised.
                client.getDREF("sim/test/test_float")
            except:
                print("Error establishing connection to X-Plane.")
                print("Exiting...")
                return
            waypoints = []
            for x in range(100):
                print('hello')
