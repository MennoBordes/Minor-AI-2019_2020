def getParameters():
    """
        This function is used to define training parameters.
        This is separated from the main loop pf the program for ease of reference.
        There are many state variables,
        so that having them in a separate file is a good idea.
    """

    globalDictionary = {
        # State Variables. This together with other parameters (to be defined later) will give us the
        # state of the aircraft. Note that this variables will be parsed to our function and the function
        # returns a set of values. check xplane dataref file for definition of stateVariable

        # Aircraft position state variable
        "stateAircraftPosition": [],
        "aircraftSpeed": "sim/flightmodel/position/indicated_airspeed",

        "headingReward": "sim/cockpit2/radios/indicators/gps_bearing_deg_mag",
        "stateVariableValue": [],

        # this is the timing data parameters from x plane DataRef. "sim/time/total_running_time_sec",
        "timer": "sim/time/total_flight_time_sec",
        "timer2": "sim/time/total_running_time_sec",
        # this is for timing data storage. It will be recovered from simulation
        "timerValue": [None],
        "timer2Value": [None],
        "on_ground": ["sim/flightmodel2/gear/on_ground"],
        "crash": ["sim/flightmodel/engine/ENGN_running"],

        "resetHold": [10.0],
        "NumOfStatesAndPositions": 14,

        "episodeReward": 0.0,
        "totalReward": 0.0,
        "flag": False,

        "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "state2": {"roll_rate": 0, "pitch_rate": 0, "altitude": 0, "pitch": 0, "roll": 0, "velocity_x": 0,
                   "velocity_y": 0, "delta_altitude": 0, "delta_heading": 0, "yaw_rate": 0},
        "episodeStep": 0,
        "reset": False,
        "elapsedTime": "sim/time/total_flight_time_sec",

        "wheelFailures": ["sim/operation/failures/rel_tire1", "sim/operation/failures/rel_tire2",
                          "sim/operation/failures/rel_tire3", "sim/operation/failures/rel_tire4",
                          "sim/operation/failures/rel_tire5"],

    }

    globalDictionary = dotdict(globalDictionary)  # Enable dot notation for dictionary

    return globalDictionary


class dotdict(dict):
    """dot (.) notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__