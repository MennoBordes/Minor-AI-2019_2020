import gym
from gym import error, spaces, utils
from gym.utils import seeding
from custom_gym.envs.myxpc.XPlaneFunctions import send_waypoints
from custom_gym.envs.myxpc import xpc2 as xpc
from custom_gym.envs.myxpc import keypress
from custom_gym.envs.myxpc.keypress import ResetXPlane
import pygetwindow
import pydirectinput
import time


class XPL(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        print("init xpl")

    def step(self, action):
        print("step")

    def reward(self):
        print("reward")
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
        # Reset airplane
        xplane_window = pygetwindow.getWindowsWithTitle("X-System")
        print(xplane_window)

    def render(self, mode="human"):
        print("render")

    def quit(self):
        print("quit")


    def test_window(self):
        
        xplane_window = pygetwindow.getWindowsWithTitle("X-System")[0]
        xplane_window.activate()
        time.sleep(4)
        keypress.PressKey(0x50)
        time.sleep(1)
        keypress.ReleaseKey(0x50)
    
    def test_keystroke(self):
        keypress.PressKey(0x50)


    def test_window2(self):
        xplane_window = pygetwindow.getWindowsWithTitle("X-System")[0]
        xplane_window.activate()
        time.sleep(6)
        pydirectinput.keyDown('p')
        time.sleep(1)
        pydirectinput.keyUp('p')

    def test_window3(self):
        xplane_window = pygetwindow.getWindowsWithTitle("X-System")[0]
        xplane_window.activate()
        time.sleep(2)
        ResetXPlane()
