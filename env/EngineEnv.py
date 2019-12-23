import random
import gym
import numpy as np

import random


AIR_DENSITY = 1.225 # mg/cc
CYL_VOLUME = 400    # cc
AFR_TARGET = 14.7
MIN_MAP = 0.3
MAX_MAP = 1.0 # atm
MIN_AFR = 8.0
MAX_AFR = 22.0
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
        self.afr = MAX_AFR
        # PPO wants symmetric action space, starts at 0 with stddev 1, i.e. [-1,1]
        # but negative action really makes no sense, so try [0,1] decigrams
        # action: squirt in dg[0,1]
        #self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        # observation: afr[8,22]
        self.observation_space = gym.spaces.Box(np.array([8]), np.array([22], dtype=np.float32))
        self.steps = 0


        #self.state = np.array([0, 0, 0]) # position, velocity, acceleration
        # action: force[-10, 10]
        #self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

        # observation: position[-10,10], velocity[-10,10], acceleration[-10,10], jerk[-10,10]
        #self.observation_space = gym.spaces.Box(np.array([-10, -10, -10, -10]), np.array([10, 10, 10, 10], dtype=np.float32))

        #self.steps = 0

    def step(self, action):
        air_mass = self.map * CYL_VOLUME * AIR_DENSITY # mg
        #fuel_mass = action[0] * 100.0 # mg
        fuel_mass = action[0] # mg
        self.afr = min(max(air_mass/(fuel_mass+1.0), MIN_AFR), MAX_AFR)

        self.steps += 1
        #done = (self.steps > 10)
        done = True # one step episode
        # MAP random walk
        #if (self.steps % 10 == 0):
        #    self.map = min(max(self.map + random.uniform(-0.2, 0.2), MIN_MAP), MAX_MAP)
        reward = 0.0 - ((self.afr - AFR_TARGET)**2)
        #if (self.steps % 10 == 0):
        print(f'map:{self.map:8.3f} air_mass:{air_mass:8.3f} fuel_mass:{fuel_mass:8.3f} afr:{self.afr:8.3f} rew:{reward:8.3f}')
        return np.array([self.afr]), reward, done, {}




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
        self.steps = 0
        return np.array([self.afr])

        #self.state = np.array([self.np_random.uniform(low=-10.0, high=10.0), 0, 0]) # position, velocity, accel
        #return np.array([self.state[0], self.state[1], self.state[2], 0])
