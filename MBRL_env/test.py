import gym
import dmbrl.env
from mbbl.env import env_register
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC


task = ['gym_pendulum', 'gym_acrobot', 'gym_invertedPendulum',  'gym_cartpole', 'gym_ant', 'gym_hopper', 'gym_reacher', 'gym_walker2d', 'gym_cheetah']
# env is the environment object compatible with Gym, env_info is a dictionary with env info
# for idx in range(len(task)):
#     env, env_info = env_register.make_env(task_name=task[idx], rand_seed=1234, misc_info={'reset_type':'gym'})
# print("Finished Type II")

env, env_info = env_register.make_env(task_name='gym_pendulum', rand_seed=1234, misc_info={'reset_type':'gym'})

model = SAC(MlpPolicy, env, verbose=1, tensorboard_log='runs/pendulum')
model.learn(total_timesteps=500, log_interval=10)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


