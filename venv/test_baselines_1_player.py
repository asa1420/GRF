"""Runs football_env on OpenAI's ppo2."""
# exact copy of run_ppo2.py from gfootball
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from absl import app
from absl import flags
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines import PPO2
from baselines.ppo2 import ppo2
import gfootball.env as football_env
from gfootball.examples import models
import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
from importlib import import_module
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger

def create_single_football_env(iprocess):
  """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name="academy_empty_goal",
      stacked=('stacked' in 'extracted_stacked'),
      rewards='scoring',
      logdir=logger.get_dir(),
      render=True)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(iprocess)), allow_early_resets=True)
  return env

def get_model_env():
    vec_env = SubprocVecEnv([
        (lambda _i=i: create_single_football_env(_i))
        for i in range(1)], context=None)
    model = ppo2.learn(
      env=vec_env,
      network='cnn',
      seed=0,
      total_timesteps=0,
      load_path='models/openai-2022-03-26-22-50-54-886228/checkpoints/00900'
  )
    return model, vec_env

def main(args):
    model, env = get_model_env()

    logger.log("Running trained model")
    obs = env.reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
    rewards = np.zeros(50)
    episode_counter = 0
    while episode_counter < 50:
        if state is not None:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            rewards[episode_counter] = episode_rew
            episode_counter +=1
            for i in np.nonzero(done)[0]:
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0
             
    average_score = numpy.mean(rewards)
    variance_in_goals = numpy.var(rewards)
    env.close()
if __name__ == '__main__':
    main(sys.argv)
