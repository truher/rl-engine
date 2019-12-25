import random
import gym
import numpy as np

import random


AIR_DENSITY = 1.225 # mg/cc
CYL_VOLUME = 400    # cc
AFR_TARGET = 14.7
FAR_TARGET = 1.0/14.7
MIN_MAP = 0.3
MAX_MAP = 1.0 # atm
MIN_AFR = 8.0
MAX_AFR = 22.0
MIN_FAR = 1.0/22.0
MAX_FAR = 1.0/8.0
LEAN_MISFIRE = 18.0
RICH_MISFIRE = 9.0


#M = 5.0
#T = 1.0
#GOAL = 0.001

# models combustion.
# each step is a combustion event
# observable state: mass of air ingested -- varies somehow.  constant to start?  square wave?
# hidden state: mass of fuel in the puddle (later)
# action: mass of fuel to inject
# reward: penalty for difference from 14.7

# TODO: calculate MAP from TPS and RPM
# TODO: fuel puddle
# TODO: better throttle schedule
# TODO: exhaust flow
# TODO: afr sensor lag
# TODO: try SAC with unscaled action?
# TODO: record fuel mass correctly, change AFR calc

class EngineEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EngineEnv, self).__init__()
        self.reward_range = (-float('inf'), 0.0)
        # state
        self.map = MIN_MAP
        #self.afr = MAX_AFR
        self.far = MAX_FAR

        # PPO
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # A2C
        #self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # but negative action really makes no sense, so try [0,1] decigrams
        # action: squirt in dg[0,1]
        #self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # SAC
        # max fuel at VE=100%, afr 11 is 44 mg.  min fuel at 30% MAP, afr 18 is 8
        #self.action_space = gym.spaces.Box(low=5.0, high=50.0, shape=(1,), dtype=np.float32)
        #self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # observation: map[0,1], afr[8,22]
        #self.observation_space = gym.spaces.Box(np.array([0.0,8.0]), np.array([1.0,22.0], dtype=np.float32))
        #self.observation_space = gym.spaces.Box(np.array([MIN_MAP, MIN_FAR]), np.array([MAX_MAP, MAX_FAR], dtype=np.float32))
        # observation is map, 1/afr - 1/afr_target
        self.observation_space = gym.spaces.Box(np.array([MIN_MAP, -0.03]), np.array([MAX_MAP, 0.06], dtype=np.float32))

        self.steps = 0

        #self.state = np.array([0, 0, 0]) # position, velocity, acceleration
        # action: force[-10, 10]
        #self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

        # observation: position[-10,10], velocity[-10,10], acceleration[-10,10], jerk[-10,10]
        #self.observation_space = gym.spaces.Box(np.array([-10, -10, -10, -10]), np.array([10, 10, 10, 10], dtype=np.float32))

        #self.steps = 0

    def observation(self):
        #return np.array([self.map, self.afr])
        # make afr symmetric
        return np.array([self.map, (self.far - FAR_TARGET)])

    # agent acts on the *previous* observation to get a reward, so change the state at the end
    def step(self, action):
        air_mass = self.map * CYL_VOLUME * AIR_DENSITY # mg
        #fuel_mass = 25.0 * (action[0] + 1.0) # mg
        fuel_mass = 25 + 25.0 * action[0]   # mg
        #fuel_mass = action[0] * 100.0 # mg
        #fuel_mass = action[0] # mg
        #self.afr = min(max(air_mass/(fuel_mass+1.0), MIN_AFR), MAX_AFR)
        self.far = min(max(fuel_mass/air_mass, MIN_FAR), MAX_FAR)

        self.steps += 1
        done = (self.steps % 10 == 0)
        #done = True
        # MAP random walk
        #if (self.steps % 10 == 0):
        #    self.map = min(max(self.map + random.uniform(-0.2, 0.2), MIN_MAP), MAX_MAP)
        #reward = 0.0 - ((self.afr - AFR_TARGET)**2)
        #reward = 0.0 - (abs(self.afr - AFR_TARGET))

        # help it to find the right AFR
        #predicted_afr = air_mass/(fuel_mass+1.0)
        predicted_far = fuel_mass/air_mass
        #reward = 0.0 - ((self.afr - AFR_TARGET)**2) - ((predicted_afr - AFR_TARGET)**2)
        #reward = 0.0 - 100.0 * ((self.far - FAR_TARGET)**2) - 100.0 * ((predicted_far - FAR_TARGET)**2)
        #reward = 0.0 - 100.0 * (abs(self.far - FAR_TARGET)) - 100.0 * (abs(predicted_far - FAR_TARGET))
        #reward = 0.0 - 1000.0 * ((self.far - FAR_TARGET)**2)
        #reward = 0.0 - 10000.0 * abs(self.far - FAR_TARGET)
        #reward = max(0.0 - 10000.0 * abs(self.far - FAR_TARGET), -200) # clamp the reward
        #reward = 0.0 - 10000.0 * ((self.far - FAR_TARGET)**2)
        reward = max(0.0 - 10000.0 * ((self.far - FAR_TARGET)**2), -5) # clamp the reward



        if (self.steps % 100 == 0):
        #if (self.steps % 1 == 0):
            afr = 1/(self.far + 0.000001)
            afr_err = afr - AFR_TARGET
            far_err = self.far - FAR_TARGET
            print(f'step:{self.steps:8d} action:{action[0]:8.3f} map:{self.map:8.3f} air_mass:{air_mass:8.3f} fuel_mass:{fuel_mass:8.3f} far:{self.far:8.4f} far_err:{far_err:8.4f} afr:{afr:8.3f} afr_err:{afr_err:8.3f} rew:{reward:8.4f}')

        # maybe change throttle position
        #if (self.steps % 100 == 0):
            #if self.map < 0.5:
            #    self.map = MAX_MAP
            #else:
            #    self.map = MIN_MAP
            #self.map = min(max(self.map + random.uniform(-0.2, 0.2), MIN_MAP), MAX_MAP)
        # scale, shape
        map_dot = (-1 if (self.steps % 2 == 0) else 1) * random.weibullvariate(0.05,1.0)
        #print(f'map_dot:{map_dot:8.3f}')

        self.map = min(max(self.map + map_dot, MIN_MAP), MAX_MAP)


        return self.observation(), reward, done, {}




#        prev_position = self.state[0]
#        prev_velocity = self.state[1]
#        prev_acceleration = self.state[2]
#        action_force = min(max(action[0], -10.0), 10.0)
#
#        next_acceleration = action_force / M
#        next_jerk = next_acceleration - prev_acceleration
#        next_velocity = prev_velocity + next_acceleration * T
#        next_position = prev_position + next_velocity * T
#
#        self.steps += 1
#        done = ((abs(next_position) < GOAL) and (abs(next_velocity) < GOAL)) or (self.steps > 100)
#        self.state = np.array([next_position, next_velocity, next_acceleration])
#        reward = 0.0 - (abs(next_position)**2) - (abs(next_velocity)**2) - (abs(next_acceleration)**2) - (abs(next_jerk)**2)
#        return np.array([next_position, next_velocity, next_acceleration, next_jerk]), reward, done, {}
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # reset the episode but not the engine
    def reset(self):
        #self.steps = 0
        return self.observation()

        #self.state = np.array([self.np_random.uniform(low=-10.0, high=10.0), 0, 0]) # position, velocity, accel
        #return np.array([self.state[0], self.state[1], self.state[2], 0])
