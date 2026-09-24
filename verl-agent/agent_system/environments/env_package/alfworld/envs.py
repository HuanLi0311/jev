# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ray

_VENDORED_PACKAGE_ROOT = os.path.dirname(__file__)
if _VENDORED_PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _VENDORED_PACKAGE_ROOT)

from alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    import torch
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()

        self.jev_weight = float(env_kwargs.get('jev_weight', 0.0)) if is_train else 0.0
        if self.jev_weight < 0:
            raise ValueError('jev_weight must be nonnegative')
        self.jev_reward_mode = env_kwargs.get('jev_reward_mode', 'trajectory_mean')
        if self.jev_reward_mode not in {'trajectory_mean', 'step_advantage', 'hindsight_step_advantage', 'hindsight_group_advantage', 'hindsight_step_only_advantage'}:
            raise ValueError('invalid jev_reward_mode')
        # Hindsight scoring runs once the complete trajectory and verifier
        # outcome exist, so this environment must not query Jev per transition.
        self.jev_enabled = self.jev_weight > 0 and not self.jev_reward_mode.startswith('hindsight_')
        if self.jev_enabled:
            from score_jev_v1 import online_alfworld_transition, parse_effect_response, query, RUBRIC_VERSION
            self.jev_transition = online_alfworld_transition
            self.jev_query = query
            self.jev_effect = parse_effect_response
            self.jev_rubric_version = RUBRIC_VERSION
            self.jev_key = os.environ.get('TYPESAFE_API_KEY')
            if not self.jev_key:
                raise ValueError('TYPESAFE_API_KEY is required for the Jev training arm')
            log_path = env_kwargs.get('jev_log_path')
            if not log_path:
                raise ValueError('jev_log_path is required for the Jev training arm')
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            # ponytail: fresh runs are protected by the launcher; append lets a
            # checkpoint resume preserve earlier online annotations.
            self.jev_log = open(log_path, 'a', encoding='utf-8')
            self.jev_rollout = -1
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.jev_enabled:
            active = [i for i in range(self.num_processes) if not self.jev_done[i]]
            states = [
                self.jev_transition(self.jev_tasks[i], self.jev_history[i],
                                    self.jev_obs[i], actions[i], text_obs_list[i])
                for i in active
            ]
            # ponytail: four concurrent calls cap API pressure; raise only after rate-limit testing.
            responses = []
            if active:
                with ThreadPoolExecutor(max_workers=min(4, len(active))) as pool:
                    responses = list(pool.map(lambda state: self.jev_query(self.jev_key, state), states))
            for i, state, response in zip(active, states, responses):
                if response.get('model') != 'jev-1.13.0':
                    raise ValueError(f"Jev model changed: {response.get('model')}")
                jev_score, jev_confidence = self.jev_effect(response)
                jev_reward = jev_confidence * (2 * jev_score - 1)
                previous_count = len(self.jev_history[i])
                previous_mean = self.jev_sum[i] / previous_count if previous_count else 0.0
                self.jev_sum[i] += jev_reward
                # ponytail: mean increments telescope, so long failed traces are not rewarded just for length.
                jev_increment = self.jev_sum[i] / (previous_count + 1) - previous_mean
                baseline_reward = rewards_list[i]
                if self.jev_reward_mode == 'trajectory_mean':
                    rewards_list[i] += self.jev_weight * jev_increment
                else:
                    # Preserve Jev's native continuous score and confidence; the
                    # step estimator combines them without group standardization.
                    info_list[i]['jev_effect_score'] = jev_score
                    info_list[i]['jev_confidence'] = jev_confidence
                self.jev_log.write(json.dumps({
                    'rollout': self.jev_rollout, 'env_index': i,
                    'step_index': len(self.jev_history[i]),
                    'rubric_version': self.jev_rubric_version,
                    'request_state': state, 'response': response,
                    'reward_mode': self.jev_reward_mode,
                    'baseline_reward': baseline_reward,
                    'jev_score': jev_score, 'jev_confidence': jev_confidence,
                    'jev_reward': jev_reward, 'jev_aggregate_increment': jev_increment,
                    'combined_reward': rewards_list[i],
                }, ensure_ascii=False) + '\n')
                self.jev_history[i].append({'action': actions[i], 'result': text_obs_list[i]})
            self.jev_log.flush()
            self.jev_obs = text_obs_list
            self.jev_done = [bool(previous or done) for previous, done in zip(self.jev_done, dones_list)]

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.jev_enabled:
            self.jev_rollout += 1
            self.jev_obs = text_obs_list
            self.jev_tasks = []
            for observation in text_obs_list:
                marker = 'Your task is to: '
                if marker not in observation:
                    raise ValueError('ALFWorld task missing from public observation')
                self.jev_tasks.append(observation.split(marker, 1)[1].strip())
            self.jev_history = [[] for _ in text_obs_list]
            self.jev_sum = [0.0 for _ in text_obs_list]
            self.jev_done = [False for _ in text_obs_list]

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)
        if self.jev_enabled:
            self.jev_log.close()

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
