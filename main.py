import gym

#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines import PPO2
from stable_baselines import SAC
from env.EngineEnv import EngineEnv

def printit(x,y):
    print(f'{x[0]:10.3f} {y:10.3f}')

# maybe less network will be less random?
class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 128, 128],
                                           layer_norm=False,
                                           feature_extraction="mlp")

def main():
    env = DummyVecEnv([lambda: EngineEnv()])
    env.env_method("seed", 0)
    #model = PPO2(MlpPolicy, env, tensorboard_log="/tmp/foo")
    model = SAC(MlpPolicy, env, tensorboard_log="/tmp/foo")
    #model = SAC(CustomSACPolicy, env, tensorboard_log="/tmp/foo")
    model.learn(total_timesteps=1000000)
    obs = env.reset()
    print('afr        reward')
    #      1234567890 1
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            printit(info[0]['terminal_observation'], rewards[0])
            print("")
        printit(obs[0], rewards[0])

if __name__ == '__main__':
    main()
