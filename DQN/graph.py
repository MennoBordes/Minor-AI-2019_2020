import matplotlib.pyplot as plt
from gym_xplane.envs import xpc2 as xpc


def draw_used_fuel(human_fuel, ai_fuel):

    # Settings x and y labels
    plt.xlabel('Agent')
    plt.ylabel('Fuel (kg)')
    # Setting graph title
    plt.title('Human fuel vs AI fuel used')

    plt.bar("Human", human_fuel, 0.2)
    plt.bar("AI", ai_fuel, 0.2)

    # Enabling labels
    plt.legend()
    # Showing the graph in a new desktop window
    plt.show()


def draw_flight_time(human_time, ai_time):
    # Settings x and y labels
    plt.xlabel('Agent')
    plt.ylabel('Time (sec)')
    # Setting graph title
    plt.title('Human time vs AI time flown')
    
    # Plotting two lines

    plt.bar("human", human_time,0.2)
    plt.bar("AI", ai_time, 0.2)

    # Enabling labels
    plt.legend()
    # Showing the graph in a new desktop window
    plt.show()


def check_time():
    print("Setting next waypoint")
    with xpc.XPlaneConnect(clientAddr='0.0.0.0', xpHost='127.0.0.1', xpPort=49009,
                           clientPort=3, timeout=3000, max_episode_steps=5000) as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return 0
        dataref = "sim/time/total_flight_time_sec"
        flight_time = client.getDREF(dataref)
        return flight_time


def check_fuel():
    print("Setting next waypoint")
    with xpc.XPlaneConnect(clientAddr='0.0.0.0', xpHost='127.0.0.1', xpPort=49009,
                           clientPort=2, timeout=3000, max_episode_steps=5000) as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return 0
        dataref = "sim/flightmodel/weight/m_fuel_total"
        fuel = client.getDREF(dataref)
        return fuel


if __name__ == '__main__':
    # Cruise:
    draw_used_fuel((86870.0390625 - 86704.7109375), (86866.890625 - 86863.765625))
    draw_flight_time((106.98492431640625 - 4.120603084564209), (20.452260971069336 - 4.673367023468018))

    # Landing
    # draw_used_fuel(1.703125, 62.6015625)
    # draw_flight_time(25.51291275024414, 28.793968200683594)

