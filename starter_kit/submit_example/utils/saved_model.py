"""
This file contains RLlib policy reload for evaluation usage, not for training.
"""
import os
import pickle
import copy

import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
import ray.rllib.agents.ppo as ppo

from smarts.core.agent import AgentPolicy
from utils.rnn_model import RNNModel
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy as LoadPolicy

import torch
tf = try_import_tf()[0]
#518 532

#########################################
# restore checkpoint generated during training
# like load_Path = "checkpoint_200/checkpoint-200"
#########################################
class RLlibTFCheckpointPolicy(AgentPolicy):
    def __init__(
        self, load_path, algorithm, policy_name, observation_space, action_space
    ):
        self._checkpoint_path = load_path
        self._algorithm = algorithm
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._sess = None

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")
        
        if self._sess:
            return

        if self._algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif self._algorithm in ["A2C", "A3C"]:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
        elif self._algorithm == "PG":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif self._algorithm == "DQN":
            from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy as LoadPolicy
        else:
            raise TypeError("Unsupport algorithm")

        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()

        with tf.name_scope(self._policy_name):
            # obs_space need to be flattened before passed to PPOTFPolicy
            flat_obs_space = self._prep.observation_space
            self.policy = LoadPolicy(flat_obs_space, self._action_space, {})
            objs = pickle.load(open(self._checkpoint_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[self._policy_name]
            self.policy.set_weights(weights)

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self.policy.compute_actions(obs, explore=False)[0]
        else:
            # single infer

            obs = self._prep.transform(obs)
            action = self.policy.compute_actions([obs], explore=False)[0][0]


        return action

class RLlibTorchCheckpointPolicy(AgentPolicy):
    def __init__(
        self, load_path, algorithm, policy_name, observation_space, action_space
    ):
        self._checkpoint_path = load_path
        self._algorithm = algorithm
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._sess = None

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        if self._sess:
            return

        if self._algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy as LoadPolicy


        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()

        with tf.name_scope(self._policy_name):
            # obs_space need to be flattened before passed to PPOTFPolicy
            flat_obs_space = self._prep.observation_space
            self.policy = LoadPolicy(flat_obs_space, self._action_space, {})
            objs = pickle.load(open(self._checkpoint_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[self._policy_name]
            self.policy.set_weights(weights)

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self.policy.compute_actions(obs, explore=False)[0]
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self.policy.compute_actions([obs], explore=False)[0][0]

        return action
#########################################
# restore checkpoint exported at end
# like load_path = "checkpoint/checkpoint"
#########################################
class RLlibFinalCkptPolicy(AgentPolicy):
    def __init__(self, load_path, observation_space, action_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._checkpoint_path = load_path

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()
        saver = tf.train.import_meta_graph(
            os.path.join(os.path.dirname(self._checkpoint_path), "model.meta")
        )
        saver.restore(
            self._sess, os.path.join(os.path.dirname(self._checkpoint_path), "model")
        )

        graph = tf.get_default_graph()

        if self.is_continuous:
            # These tensor names were found by inspecting the trained model
            # deterministic
            self.output_node = graph.get_tensor_by_name("default_policy/split:0")
            # add guassian noise
            # output_node = graph.get_tensor_by_name("default_policy/add:0")
        else:
            self.output_node = graph.get_tensor_by_name("default_policy/ArgMax:0")

        self.input_node = graph.get_tensor_by_name("default_policy/observation:0")

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs.values()]
            action = self._sess.run(self.output_node, feed_dict={self.input_node: obs})
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self._sess.run(
                self.output_node, feed_dict={self.input_node: [obs]}
            )[0]

        return action


#########################################
# restore model exported at end
# like load_path = "model"
#########################################
class RLlibTFModelPolicy(AgentPolicy):
    def __init__(self, load_path, observation_space, action_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._path_to_model = load_path

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()
        tf.saved_model.load(self._sess, export_dir=self._path_to_model, tags=["serve"])

        graph = tf.get_default_graph()

        if self.is_continuous:
            # These tensor names were found by inspecting the trained model
            # deterministic
            self.output_node = graph.get_tensor_by_name("default_policy/split:0")
            # add guassian noise
            # output_node = graph.get_tensor_by_name("default_policy/add:0")
        else:
            self.output_node = graph.get_tensor_by_name("default_policy/ArgMax:0")

        self.input_node = graph.get_tensor_by_name("default_policy/observation:0")

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self._sess.run(self.output_node, feed_dict={self.input_node: obs})
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self._sess.run(
                self.output_node, feed_dict={self.input_node: [obs]}
            )[0]

        return action

#########################################
# torch rnn model
# 
#########################################
class RLlibTorchGRUPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        flat_obs_space = self._prep.observation_space

        ray.init(ignore_reinit_error=True, local_mode=True)

        
        
        ModelCatalog.register_custom_model("my_rnn", RNNModel)
        config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        config["model"]["custom_model"] = "my_rnn"

        self.policy = LoadPolicy(flat_obs_space, self._action_space, config)
        objs = pickle.load(open(self._checkpoint_path, "rb"))
        objs = pickle.loads(objs["worker"])
        state = objs["state"]
        filters = objs["filters"]
        self.filters = filters[self._policy_name]
        weights = state[self._policy_name]
        weights.pop("_optimizer_variables")
        self.policy.set_weights(weights)
        self.model = self.policy.model

        self.rnn_state = self.model.get_initial_state()
        self.rnn_state = [torch.reshape(self.rnn_state[0], shape=(1, -1))]

    def act(self, obs):

        # single infer
        obs = self._prep.transform(obs)
        obs = self.filters(obs, update=False)
        action, self.rnn_state, _ = self.policy.compute_actions([obs], self.rnn_state, explore=False)
        action = action[0]

        return action
