from copy import deepcopy

import torch
from torch import nn

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from model import MyModelClass, TargetPredictor
from wrappers import *


config = {
    "clip_param": 0.2,
    "entropy_coeff": 0.01,
    "env": "my_env",
    "framework": "torch",
    "lambda": 0.95,
    "logger_config": {
    "wandb": {
      "name": "PPO MultiTask (C3, C17, C32, C8) pretrained (AngelaCNN) (3 noops after placement) r: -0.01 div10",
      "project": "IGLU-Minecraft"
    }
    },
    "model": {
    "custom_model": "my_torch_model",
    "custom_model_config": {}
    },
    "num_gpus": 1,
    "num_workers": 1,
    "sgd_minibatch_size": 256,
    "train_batch_size": 1000
}

def fake_env_creator(env_config=None):
    env = FakeIglu({})
    #env = VisualObservationWrapper(env, include_target=True)
    env = SelectAndPlace(env)
    env = Discretization(env, flat_action_space('human-level'))
    #env = RewardWrapper(env)
    return env

class CustomAgent:
    def __init__(self, action_space):
        ray.init(local_mode=True)
        ModelCatalog.register_custom_model("my_torch_model", MyModelClass)
        register_env("my_fake_env", fake_env_creator)
        self.fake_env = fake_env_creator()
        trainer = PPOTrainer(config=config, env="my_fake_env")
        trainer.restore("checkpoint/checkpoint-510")
        self.action_space = action_space
        self.target_predictor = TargetPredictor()
        self.agent = trainer
        self.actions = iter([])
        self.target_grid = None
        self.state = None
        self.step = 0

    def policy(self, obs, reward, done, info, state):
        #if self.target_grid is None:
        self.target_grid = self.target_predictor(obs["chat"]).detach().numpy()
        del obs["chat"]
        obs = {
            'pov': obs['pov'].astype(np.float32),
            'inventory': obs['inventory'],
            'compass': np.array([obs['compass']['angle'].item()]),            
            'target_grid': self.target_grid
        }
        output = self.agent.compute_single_action(
            obs, explore=False, state=state
        )
        if not isinstance(output, tuple):
            action = output
        else:
            action, state, _ = output
        return action, state

    def act(self, obs, reward, done, info):
        if done:
            self.actions = iter([])
            self.state = None
            self.step = 0
            return

        try:
            action = next(self.actions)
        except StopIteration:
            #obs = self.fake_env.wrap_observation(obs, reward, done, info)
            agent_action, self.state = self.policy(obs, reward, done, info, self.state)
            self.actions = iter(self.fake_env.stack_actions()(agent_action))
            action = next(self.actions)     

        self.step += 1
        return deepcopy(action)
