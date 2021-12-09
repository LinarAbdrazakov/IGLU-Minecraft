from threading import stack_size
import gym
import os
import cv2
import shutil
import datetime
import pickle
import json
import uuid
import logging
from gym.core import ActionWrapper
import numpy as np
from collections import defaultdict
from typing import Generator
from minerl_patched.herobraine.hero import spaces

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')


class Wrapper(gym.Wrapper):
    def stack_actions(self):
        if isinstance(self.env, Wrapper):
            return self.env.stack_actions()

    def wrap_observation(self, obs, reward, done, info):
        if hasattr(self.env, 'wrap_observation'):
            return self.env.wrap_observation(obs, reward, done, info)
        else:
            return obs


class ActionsWrapper(Wrapper):
    def wrap_action(self, action) -> Generator:
        raise NotImplementedError

    def stack_actions(self):
        def gen_actions(action):
            for action in self.wrap_action(action):
                wrapped = None
                if hasattr(self.env, 'stack_actions'):
                    wrapped = self.env.stack_actions()
                if wrapped is not None:
                    yield from wrapped(action)
                else:
                    yield action
        return gen_actions

    def step(self, action):
        total_reward = 0
        for a in self.wrap_action(action):
            obs, reward, done, info = super().step(a)
            total_reward += reward
            if done: 
                return obs, total_reward, done, info
        return obs, total_reward, done, info


class ObsWrapper(Wrapper):
    def observation(self, obs, reward=None, done=None, info=None):
        raise NotImplementedError

    def wrap_observation(self, obs, reward, done, info):
        new_obs = self.observation(obs, reward, done, info)
        return self.env.wrap_observation(new_obs, reward, done, info)

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.observation(obs, reward, done, info), reward, done, info


class TimeLimit(Wrapper):
    def __init__(self, env, limit):
        super().__init__(env)
        self.limit = limit
        self.step_no = 0

    def reset(self):
        self.step_no = 0
        return super().reset()

    def step(self, action):
        self.step_no += 1
        obs, reward, done, info = super().step(action)
        if self.step_no >= self.limit:
            done = True
        return obs, reward, done, info



class SizeReward(Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.size = 0

  def reset(self):
    self.size = 0
    return super().reset()

  def step(self, action):
    obs, reward, done, info = super().step(action)
    intersection = self.env.unwrapped.task.task_monitor.max_int
    reward = max(intersection, self.size) - self.size
    self.size = max(intersection, self.size)
    return obs, reward, done, info


class SelectAndPlace(ActionsWrapper):
    def wrap_action(self, action):
        if action['hotbar'] != 0:
            yield action
            action = self.env.action_space.noop()
            action['use'] = 1
        if action['use'] == 1 or action['attack'] == 1:
            for _ in range(3):
                yield action
                action = self.env.action_space.noop()
        yield action


def flat_action_space(action_space):
    if action_space == 'human-level':
        return flat_human_level
    if action_space == 'discrete':
        return flat_discrete


def flat_human_level(env, camera_delta=5):
    binary = ['attack', 'forward', 'back', 'left', 'right', 'jump']
    discretes = [env.action_space.no_op()]
    for op in binary:
        dummy = env.action_space.no_op()
        dummy[op] = 1
        discretes.append(dummy)
    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    for i in range(6):
        dummy = env.action_space.no_op()
        dummy['hotbar'] = i + 1
        discretes.append(dummy)
    discretes.append(env.action_space.no_op())
    return discretes


def flat_discrete(env, camera_delta=5):
    discretes = [env.action_space.no_op()]
        
    forward = env.action_space.no_op()
    forward['forward'] = 2
    discretes.append(forward)
    backward = env.action_space.no_op()
    backward['forward'] = 1
    discretes.append(backward)
    
    left = env.action_space.no_op()
    left['strafe'] = 1
    discretes.append(left)
    right = env.action_space.no_op()
    right['strafe'] = 2
    discretes.append(right)

    jumpforward = env.action_space.no_op()
    jumpforward['forward'] = 2
    jumpforward['jump'] = 1
    discretes.append(jumpforward)
    jumpbackward = env.action_space.no_op()
    jumpbackward['forward'] = 1
    jumpbackward['jump'] = 1
    discretes.append(jumpbackward)
    
    jumpleft = env.action_space.no_op()
    jumpleft['strafe'] = 1
    jumpleft['jump'] = 1
    discretes.append(jumpleft)
    jumpright = env.action_space.no_op()
    jumpright['strafe'] = 2
    jumpright['jump'] = 1
    discretes.append(jumpright)

    attack = env.action_space.no_op()
    attack['attack'] = 1
    discretes.append(attack)

    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    for i in range(6):
        dummy = env.action_space.no_op()
        dummy['hotbar'] = i + 1
        discretes.append(dummy)
    return discretes


class Discretization(ActionsWrapper):
    def __init__(self, env, flatten):
        super().__init__(env)
        camera_delta = 5
        self.discretes = flatten(env, camera_delta)
        self.action_space = gym.spaces.Discrete(len(self.discretes))
        self.old_action_space = env.action_space
        self.last_action = None

    def wrap_action(self, action=None, raw_action=None):
        if action is not None:
            action = self.discretes[action]
        elif raw_action is not None:
            action = raw_action
        yield action


class FakeIglu(gym.Env):
    def __init__(self, config, wrap_actions=True):
        self.action_space = spaces.Dict({
            'forward': spaces.Discrete(2),
            'back': spaces.Discrete(2),
            'left': spaces.Discrete(2),
            'right': spaces.Discrete(2),
            'jump': spaces.Discrete(2),
            'camera': spaces.Box(low=-180.0, high=180.0, shape=(2,)),
            'attack': spaces.Discrete(2),
            'use': spaces.Discrete(2),
            'hotbar': spaces.Discrete(7),
        })
        
        self.observation_space = spaces.Dict({
            'pov': spaces.Box(0, 255, (64, 64, 3), dtype=np.float32),
            'inventory': spaces.Box(low=0, high=20, shape=(6,), dtype=np.float32),
            'compass': spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            'target_grid': spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11))
        })
    
        self.step = 0

    def reset(self):
        self.step = 0 
        return self.observation_space.sample()
  
    def step(self, action):
        self.step += 1
        done = self.step >= 1000
        reward = 0
        info = {}
        return self.observation_space.sample(), reward, done, info

    def update_taskset(self, *args, **kwargs):
        pass

    def set_task(self, *args, **kwargs):
        pass


class VideoLogger(Wrapper):
    def __init__(self, env, every=50):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'action_logs/run-{runtime}'
        self.every = every
        self.filename = None
        self.running_reward = 0
        self.actions = []
        self.flushed = False
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            with open(f'{self.filename}-r{self.running_reward}.json', 'w') as f:
                json.dump(self.actions, f)
            self.out.release()
            with open(f'{self.filename}-obs.pkl', 'wb') as f:
                pickle.dump(self.obs, f)
            self.obs = []
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        uid = str(uuid.uuid4().hex)
        name = f'episode-{timestamp}-{uid}'
        self.filename = os.path.join(self.dirname, name)
        self.running_reward = 0
        self.flushed = True
        self.actions = []
        self.frames = []
        self.obs = []
        self.out = cv2.VideoWriter(f'{self.filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   20, (64, 64))

    def reset(self):
        if not self.flushed:
            self.flush()
        return super().reset()

    def close(self):
        if not self.flushed:
            self.flush()
        return super().close()

    def step(self, action):
        # assuming dict
        self.flushed = False
        new_action = {}
        for key in action:
            new_action[key] = action[key]
            if isinstance(new_action[key], np.ndarray):
                new_action[key] = new_action[key].tolist()
        obs, reward, done, info = super().step(action)
        self.actions.append(new_action)
        self.out.write(obs['pov'][..., ::-1])
        self.obs.append({k: v for k, v in obs.items() if k != 'pov'})
        self.obs[-1]['reward'] = reward
        self.running_reward += reward
        return obs, reward, done, info


class Logger(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'action_logs/run-{runtime}'
        self.filename = None
        self.running_reward = 0
        self.actions = []
        self.flushed = False
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            with open(f'{self.filename}-r{self.running_reward}.json', 'w') as f:
                json.dump(self.actions, f)
            self.out.release()
            with open(f'{self.filename}-obs.pkl', 'wb') as f:
                pickle.dump(self.obs, f)
            self.obs = []
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        uid = str(uuid.uuid4().hex)
        name = f'episode-{timestamp}-{uid}'
        self.filename = os.path.join(self.dirname, name)
        self.running_reward = 0
        self.flushed = True
        self.actions = []
        self.frames = []
        self.obs = []
        self.out = cv2.VideoWriter(f'{self.filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   20, (64, 64))

    def reset(self):
        if not self.flushed:
            self.flush()
        return super().reset()

    def close(self):
        if not self.flushed:
            self.flush()
        return super().close()

    def step(self, action):
        # assuming dict
        self.flushed = False
        new_action = {}
        for key in action:
            new_action[key] = action[key]
            if isinstance(new_action[key], np.ndarray):
                new_action[key] = new_action[key].tolist()
        obs, reward, done, info = super().step(action)
        self.actions.append(new_action)
        self.out.write(obs['pov'][..., ::-1])
        self.obs.append({k: v for k, v in obs.items() if k != 'pov'})
        self.obs[-1]['reward'] = reward
        self.running_reward += reward
        return obs, reward, done, info


class VisualObservationWrapper(ObsWrapper):
    def __init__(self, env, include_target=False):
        super().__init__(env)
        self.observation_space = {   
            'pov': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3)),
            'inventory': gym.spaces.Box(low=0.0, high=20.0, shape=(6,)),
            'compass': gym.spaces.Box(low=-180.0, high=180.0, shape=(1,)),
            'target_grid': gym.spaces.Box(low=0, high=6, shape=(9, 11, 11))
        }
        self.observation_space = gym.spaces.Dict(self.observation_space)

    def observation(self, obs, reward=None, done=None, info=None):
        if info is not None:
            if 'target_grid' in info:
                target_grid = info['target_grid']
                del info['target_grid']
            else:
                logger.error(f'info: {info}')
                if hasattr(self.unwrapped, 'should_reset'):
                    self.unwrapped.should_reset(True)
                target_grid = np.zeros((9, 11, 11))
        else:
            target_grid = np.zeros((9, 11, 11))
        return {
            'pov': obs['pov'].astype(np.float32),
            'inventory': obs['inventory'],
            'compass': np.array([obs['compass']['angle'].item()]),            
            'target_grid': target_grid
        }
    

class VectorObservationWrapper(ObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            'agentPos': gym.spaces.Box(low=-5000.0, high=5000.0, shape=(5,)),
            'grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11)),
            'inventory': gym.spaces.Box(low=0.0, high=20.0, shape=(6,)),
            'target_grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11))
        })

    def observation(self, obs, reward=None, done=None, info=None):
        if IGLU_ENABLE_LOG == '1':
            self.check_component(
                obs['agentPos'], 'agentPos', self.observation_space['agentPos'].low,
                self.observation_space['agentPos'].high
            )
            self.check_component(
                obs['inventory'], 'inventory', self.observation_space['inventory'].low,
                self.observation_space['inventory'].high
            )
            self.check_component(
                obs['grid'], 'grid', self.observation_space['grid'].low,
                self.observation_space['grid'].high
            )
        if info is not None:
            if 'target_grid' in info:
                target_grid = info['target_grid']
                del info['target_grid']
            else:
                logger.error(f'info: {info}')
                if hasattr(self.unwrapped, 'should_reset'):
                    self.unwrapped.should_reset(True)
                target_grid = self.env.unwrapped.tasks.current.target_grid
        else:
            target_grid = self.env.unwrapped.tasks.current.target_grid
        return {
            'agentPos': obs['agentPos'],
            'inventory': obs['inventory'],
            'grid': obs['grid'],
            'target_grid': target_grid
        }

    def check_component(self, arr, name, low, hi):
        if (arr < low).any():
            logger.info(f'{name} is below level {low}:')
            logger.info((arr < low).nonzero())
            logger.info(arr[arr < low])
        if (arr > hi).any():
            logger.info(f'{name} is above level {hi}:')
            logger.info((arr > hi).nonzero())
            logger.info(arr[arr > hi])
