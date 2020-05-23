import gym
from time import sleep
import numpy as np
import gym_xplane.parameters as params

import gym_xplane.envs.xpc as xpc
from gym import error, spaces, utils
from gym.utils import seeding


class XplaneENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, client):
        self.value = "not yet implemented"
        self.client = client
        self.action_space = spaces.Dict({"Elevons": spaces.Box(low=-1, high=1, shape=()),
                                         "Ailerons": spaces.Box(low=-1, high=1, shape=()),
                                         "Rudder": spaces.Box(low=-1, high=1, shape=()),
                                         "Left Throttle": spaces.Box(low=-1, high=1, shape=()),
                                         "Right Throttle": spaces.Box(low=-1, high=1, shape=()),
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


    def step(self, client, actions):
        self.parameters.flag = False
        self.parameters.episodeReward = 0
        self.episode_steps += 1
        self.client = client
        tempreward = 0.
        tempreward2 = 0.
        tempreward3 = 0.
        perturbationAllowed = 15.0
        minimumAltitude = 550  # meters
        minimumRuntime = 80.50
        minimumDistance = 0.5;

        try:
            act = [actions['Elevons']]

            act.extend(
                [actions['Ailerons'], actions['Rudder'], actions['Left Throttle'], actions['Right Throttle'], int(actions['Gear']),
                 actions['Flaps']])

            self.client.sendCTRL(act)

            state = [];

            stateVariableTemp = self.client.getDREFs(self.parameters.stateVariable)
            self.parameters.stateAircraftPosition = list(self.client.getPOSI());

            self.parameters.stateVariableValue = [i[0] for i in stateVariableTemp]

            # 14 prameters
            state = self.parameters.stateAircraftPosition + self.parameters.stateVariableValue

            if len(state) == 14:
                self.parameters.state14 = state
            else:
                self.parameters.state14 = self.parameters.state14

            rewardVector = self.client.getDREF(self.parameters.rewardVariable)
            headingReward = self.client.getDREF(self.parameters.headingReward)[0][0]

            if self.parameters.stateAircraftPosition[5] > headingReward + perturbationAllowed:
                tempreward -= 1.

            else:
                tempreward += 1.

            if self.parameters.stateAircraftPosition[5] < minimumAltitude:
                tempreward2 -= 1.
            else:
                tempreward2 += 1.

            if rewardVector[0][0] > minimumDistance:
                tempreward3 -= 0.5
            else:
                tempreward3 += 0.5

            self.parameters.episodeReward = (tempreward + tempreward2 + tempreward3) / 3.

            self.parameters.totalReward += self.parameters.episodeReward

            if self.client.getDREFs(self.parameters.on_ground)[0][0] >= 1 or \
                    client.getDREFs(self.parameters.crash)[0][0] <= 0:

                self.parameters.flag = True
                self.parameters.totalReward -= 2


            elif self.client.getDREF(self.parameters.timer2)[0][0] > minimumRuntime:
                self.parameters.flag = True
                self.parameters.totalReward += 2

            # ***************** reformat and send  action ***************************************************

            if self.episode_steps >= self.max_episode_steps:
                self.parameters.flag = True
                reward = self.parameters.totalReward

            ### final episode or loop episode
            if self.parameters.flag:
                reward = self.parameters.totalReward
                # print('final reward', self.parameters.totalReward )
                self.parameters.flag = True
                self.parameters.totalReward = 0.
            else:
                reward = self.parameters.episodeReward
                # print('total before',reward  )
            # agent.observe(state, actions);
            # print('flag; ',parameters.flag,' episode reward: ',parameters.episodeReward,
            #               ' Total reward',parameters.totalReward, ' reward:',reward)
            # print('reward table',self.parameters.episodeReward,self.parameters.totalReward)

        except:
            reward = self.parameters.episodeReward
            self.parameters.flag = False
            self.parameters.state14 = self.parameters.state14

        return self.parameters.state14, reward, self.parameters.flag, self._get_info()
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

    def render(self, mode='human', close=False):
        pass
