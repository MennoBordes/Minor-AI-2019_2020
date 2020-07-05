import matplotlib.pyplot as plt
from custom_gym.envs.myxpc import xpc2 as xpc

def draw_used_fuel(n_waypoints, human_fuel, ai_fuel):
    # Settings x and y labels
    plt.xlabel('Waypoint')
    plt.ylabel('Fuel')
    # Setting graph title
    plt.title('Human fuel vs AI fuel used')
    
    # Plotting two lines
    x = [0,n_waypoints]
    print(x)
    plt.plot(x, [0, human_fuel],'r', label = "Human fuel")
    plt.plot(x, [0, ai_fuel], 'b', label = "AI Fuel")
    # Enabling labels
    plt.legend()
    # Showing the graph in a new desktop window
    plt.show()


def draw_flight_time(n_waypoints, human_time, ai_time):
    # Settings x and y labels
    plt.xlabel('Waypoint')
    plt.ylabel('Time')
    # Setting graph title
    plt.title('Human time vs AI time flown')
    
    # Plotting two lines
    x = [0,n_waypoints]
    print(x)
    plt.plot(x, [0, human_time],'g', label = "Human time")
    plt.plot(x, [0, ai_time], 'y', label = "AI time")
    # Enabling labels
    plt.legend()
    # Showing the graph in a new desktop window
    plt.show()


def check_time():
    print("Setting next waypoint")
    with xpc.XPlaneConnect() as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
        dataref = "sim/time/total_flight_time_sec"
        flight_time = client.getDREF(dataref)
        print(flight_time)


def check_fuel():
    print("Setting next waypoint")
    with xpc.XPlaneConnect() as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
        dataref = "sim/flightmodel/weight/m_fuel_total"
        fuel = client.getDREF(dataref)
        print(fuel)
    
def bar_graph_fuel(human_fuel, ai_fuel):
    plt.title('Human fuel vs AI fuel used')
    plt.xlabel('Agent')
    plt.ylabel('Fuel (kg)')
    plt.bar("Human", human_fuel, 0.2)
    plt.bar("AI", ai_fuel, 0.2)
    plt.show()

def bar_graph_time(human_time, ai_time):
    plt.title('Human time vs AI time flown')
    plt.xlabel('Agent')
    plt.ylabel('Time (sec)')
    plt.bar('Human', human_time, 0.2)
    plt.bar('AI', ai_time, 0.2)
    plt.show()


# bar_graph_fuel(168.529296875, 130.5126953125)
# bar_graph_time(108.01172637939453, 111.30653381347656)