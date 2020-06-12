from time import sleep
from custom_gym.envs.myxpc import xpc2 as xpc
import json


# def get_state():

# def get_reward():

def get_waypoints():
    file = open("customGym/custom_gym/envs/myxpc/EHAM_EDDH.json")
    data = json.load(file)
    wp = []
    for points in data['nodes']:
        latitude = points['lat']
        longitude = points['lon']
        altitude = points['alt']
        wp.append([latitude,longitude,altitude])
    wp
    print(wp)
    



def test_waypoint():
    print("X-Plane Connect example script")
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
        client.sendWYPT(1,[52.308099999999996,4.764170000000007, 0, 53.07139999999998,7.195830000000001,74146.982,53.633700000000005,9.985260000000011,0])
        print('sent')




