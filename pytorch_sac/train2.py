#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import random
import gym
import dmbrl.env
from mbbl.env import env_register
import tensorflow as tf

import hydra
from copy import deepcopy
from scipy.io import savemat


rets, rets_ste, num_steps = [], [], []

tasks_dict = [{'mujoco': True, 'env': 'MBRLHalfCheetah-v0'},
              {'mujoco': True, 'env': 'MBRLReacher3D-v0'},
              {'mujoco': False, 'env': 'gym_pendulum'},
              {'mujoco': False, 'env': 'gym_acrobot'},
              {'mujoco': False, 'env': 'gym_invertedPendulum'},
              {'mujoco': False, 'env': 'gym_ant'}]


sac_agents = ['sac_half_cheetah', 'sac_reacher3d', 'sac_pendulum', 'sac_acrobot',
              'sac_inverted_pendulum', 'sac_ant']

WORK_DIR = None


def save_eval_mat(log_name):
    savemat(
        WORK_DIR + '/' + log_name + ".mat",
        {
            "test_returns": rets,
            "test_returns_std_err": rets_ste,
            'num_steps': num_steps
        }
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed)
    torch.manual_seed(seed)
    print("Using seed {}".format(seed))


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if tasks_dict[cfg.taskid-1]['mujoco']:
        env = gym.make(tasks_dict[cfg.taskid-1]['env'])
    else:
        env, _ = env_register.make_env(tasks_dict[cfg.taskid-1]['env'], rand_seed=1234, misc_info={'reset_type': 'gym'})
    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        global WORK_DIR
        WORK_DIR = self.work_dir
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)
        training_seed = cfg.seed
        print('env {}, seed {}'.format(cfg.env, cfg.seed))
        set_seed(training_seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)
        self.env.seed(1234)
        self.eval_env = deepcopy(self.env)
        self.eval_env.seed(0)

        cfg.agent.params.obs_dim = int(self.env.observation_space.shape[0])
        cfg.agent.params.action_dim = int(self.env.action_space.shape[0])
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        test_rets = []
        steps_in_this_episode = []
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.eval_env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            i_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.eval_env.step(action)
                i_step += 1
                # self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            test_rets.append(average_episode_reward)
            steps_in_this_episode.append(i_step)
            # self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        rets.append(np.mean(test_rets))
        rets_ste.append(np.std(test_rets) / np.sqrt(5))
        num_steps.append(steps_in_this_episode)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

        save_eval_mat(self.cfg.experiment)


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
