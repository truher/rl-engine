import gym

from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.common.vec_env import VecNormalize

# SAC
#from stable_baselines.sac.policies import FeedForwardPolicy
#from stable_baselines.sac.policies import MlpPolicy
#from stable_baselines import SAC

# PPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

# A2C
#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines import A2C

from env.EngineEnv import EngineEnv

def printit(x,y):
    map = x[0]
    far_err = x[1]
    far = far_err + 1.0/14.7
    afr = 1/far
    print(f'{x[0]:10.3f} {x[1]:10.5f} {afr:10.3f} {y:10.4f}')

# maybe less network will be less random?
#class CustomSACPolicy(FeedForwardPolicy):
#    def __init__(self, *args, **kwargs):
#        super(CustomSACPolicy, self).__init__(*args, **kwargs,
#                                           layers=[128, 128, 128],
#                                           layer_norm=False,
#                                           feature_extraction="mlp")

def main():
    env = DummyVecEnv([lambda: EngineEnv()])
    #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    #env = VecNormalize(env)
    env.env_method("seed", 0)
    # more value function loss
    model = PPO2(MlpPolicy, env, vf_coef=10.0, tensorboard_log="/tmp/foo")
    #model = A2C(MlpPolicy, env, tensorboard_log="/tmp/foo")
    #model = SAC(MlpPolicy, env, tensorboard_log="/tmp/foo")
    #model = SAC(CustomSACPolicy, env, tensorboard_log="/tmp/foo")
    model.learn(total_timesteps=1000000)
    #model.learn(total_timesteps=400000)
    #model.learn(total_timesteps=500000)
    obs = env.reset()
    print('map        far_err    afr        reward')
    #      1234567890 1234567890 1234567890 1
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #if done:
        #    printit(info[0]['terminal_observation'], rewards[0])
        #    print("")
        printit(obs[0], rewards[0])

if __name__ == '__main__':
    main()
