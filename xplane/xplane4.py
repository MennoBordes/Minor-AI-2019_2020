import XPlaneConnect.xpc as xpc

with xpc.XPlaneConnect() as client:
    dref = "sim/cockpit/switches/gear_handle_status"
    value = client.getDREF(dref)
    print("The gear handle status is " + str(value[0]))
    client.pauseSim(True)