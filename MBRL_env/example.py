import gym
import dmbrl.env
task = ['MBRLHalfCheetah-v0', 'MBRLReacher3D-v0']
for idx in range(len(task)):
    env = gym.make(task[idx])
print("Finished Type I")

from mbbl.env import env_register
task = ['gym_pendulum', 'gym_acrobot', 'gym_invertedPendulum',  'gym_cartpole', 'gym_ant', 'gym_hopper', 'gym_reacher', 'gym_walker2d', 'gym_cheetah']
# env is the environment object compatible with Gym, env_info is a dictionary with env info
for idx in range(len(task)):
    env, env_info = env_register.make_env(task_name=task[idx], rand_seed=1234, misc_info={'reset_type':'gym'})
print("Finished Type II")