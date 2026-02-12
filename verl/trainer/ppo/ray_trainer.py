# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional
from pprint import pprint
import numpy as np
import ray
import requests
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.trainer.ppo.fuse_helper import get_numeric_score, normalize_is_following_list
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask, masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.reward_score.if_score.reward_compute import compute_score as if_score_compute
@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

@dataclass
class RewriteJobSpec:
    uid: str
    template_idx: int
    candidate_idx: int
    rewrite_prompt: str
    best_original_score: float
    data_source: Any
    reward_model: Any
    extra_info: Any
    rollout_reward_scores: Any
    rewrite_suffix: int


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        reward_extra_infos_dict.pop("is_following_list", None)  # remove this key if exists
        reward_extra_infos_dict.pop("constraint_results", None)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _pad_or_trim_prompts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_length: int):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        seq_len = input_ids.size(1)
        if seq_len > target_length:
            input_ids = input_ids[:, -target_length:]
            attention_mask = attention_mask[:, -target_length:]
        elif seq_len < target_length:
            pad_len = target_length - seq_len
            pad_tokens = torch.full((input_ids.size(0), pad_len), pad_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((attention_mask.size(0), pad_len), dtype=attention_mask.dtype)
            input_ids = torch.cat([pad_tokens, input_ids], dim=1)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
        return input_ids, attention_mask

    def _decode_prompts_and_responses(self, batch: DataProto) -> tuple[list[str], list[str]]:
        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        prompt_len = prompt_ids.shape[1]
        response_len = response_ids.shape[1]
        prompts, responses = [], []
        for i in range(len(batch)):
            valid_prompt = int(attention_mask[i, :prompt_len].sum().item())
            prompt_tokens = prompt_ids[i, -valid_prompt:]
            prompts.append(self.tokenizer.decode(prompt_tokens, skip_special_tokens=True))
            valid_response = int(attention_mask[i, prompt_len : prompt_len + response_len].sum().item())
            response_tokens = response_ids[i, :valid_response]
            responses.append(self.tokenizer.decode(response_tokens, skip_special_tokens=True))
        return prompts, responses

    def _materialize_reward_infos(self, reward_extra_infos: dict[str, list] | None, batch_size: int) -> list[dict]:
        reward_infos = [dict() for _ in range(batch_size)]
        if not reward_extra_infos:
            return reward_infos
        for key, values in reward_extra_infos.items():
            if isinstance(values, np.ndarray):
                values = values.tolist()
            assert len(values) == batch_size, f"{key} length mismatches batch size"
            for idx, val in enumerate(values):
                reward_infos[idx][key] = val
        return reward_infos

    def _extract_constraints_for_prompt(
        self,
        score: Any,
        fallback_is_list: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        constraints: list[dict[str, Any]] = []
        if isinstance(score, dict):
            constraint_results = score.get("constraint_results") or []
            if isinstance(constraint_results, list):
                for idx, item in enumerate(constraint_results, start=1):
                    desc = item.get("description")
                    if not desc:
                        instr_id = item.get("instruction_id") or item.get("id")
                        if instr_id:
                            if isinstance(instr_id, (list, tuple)):
                                desc = " / ".join(map(str, instr_id))
                            else:
                                desc = str(instr_id)
                    constraints.append(
                        {
                            "idx": idx,
                            "description": desc or f"Constraint #{idx}",
                            "is_following": bool(item.get("is_following")),
                        }
                    )
        if not constraints and fallback_is_list:
            for idx, flag in enumerate(fallback_is_list, start=1):
                constraints.append(
                    {
                        "idx": idx,
                        "description": f"Constraint #{idx}",
                        "is_following": bool(flag),
                    }
                )
        return constraints

    def _extract_rubrics_from_extra_info(self, extra_info: Any) -> list[str]:
        if not extra_info or not isinstance(extra_info, dict):
            return []
        rubrics = extra_info.get("rubric_list")
        if rubrics is None:
            rubrics = extra_info.get("rubric")
        if isinstance(rubrics, str):
            try:
                parsed = json.loads(rubrics)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                rubrics = parsed
        if not isinstance(rubrics, list):
            rubrics = [rubrics] if rubrics is not None else []
        results: list[str] = []
        for item in rubrics:
            if isinstance(item, dict):
                desc = item.get("description") or item.get("title") or json.dumps(item, ensure_ascii=True)
                results.append(str(desc))
            elif item is not None:
                results.append(str(item))
        return results

    def _build_self_rewrite_prompt(
        self,
        instruction: str,
        candidate: dict[str, Any],
        target_score: float = 1.0,
        prompt_mode: str = "constraint",
    ) -> str:
        """
        Build a rewrite prompt for either constraint-based or rubric-based feedback.
        """
        prompt_mode = (prompt_mode or "IF_hir").lower()
        if prompt_mode in ("medical_rar"):
            rubric_block = "none"
            satisfied_block = None
            unmet_block = None
            unclear_block = None

            constraint_results = candidate.get("constraint_results")
            if isinstance(constraint_results, list) and constraint_results:
                satisfied: list[str] = []
                unmet: list[str] = []
                unclear: list[str] = []
                all_rubrics: list[str] = []
                for item in constraint_results:
                    if not isinstance(item, dict):
                        continue
                    desc = (item.get("description") or item.get("title") or "").strip()
                    if not desc:
                        continue
                    weight = item.get("weight")
                    weight_text = ""
                    if isinstance(weight, (int, float)):
                        weight_text = f" (weight: {weight:g})"
                    line = f"- {desc}{weight_text}"
                    all_rubrics.append(line)
                    if item.get("is_following") is True:
                        satisfied.append(line)
                    elif item.get("is_following") is False:
                        unmet.append(line)
                    else:
                        unclear.append(line)
                if all_rubrics:
                    rubric_block = "\n".join(all_rubrics)
                satisfied_block = "\n".join(satisfied) if satisfied else "none"
                unmet_block = "\n".join(unmet) if unmet else "none"
                unclear_block = "\n".join(unclear) if unclear else "none"
            else:
                rubrics = candidate.get("rubrics", [])
                rubric_block = "\n".join(f"- {r}" for r in rubrics) if rubrics else "none"

            status_block = ""
            if satisfied_block is not None and unmet_block is not None:
                status_block = f"""
[Rubric Status]
Satisfied:
{satisfied_block}

Not satisfied:
{unmet_block}

Unclear:
{unclear_block}
"""

            prompt = f"""
You are a rewrite assistant. Keep satisfied rubrics intact and fix the unmet ones. Avoid adding new facts.

[Instruction]
{instruction}

{status_block}

[Candidate]
Draft:
{candidate.get("response", "")}

Return only the rewritten answer that satisfies the rubrics."""
            return prompt.strip()

        constraints = candidate.get("constraints", [])
        satisfied_desc = [f"{c['idx']}. {c['description']}" for c in constraints if c["is_following"]]
        violated_desc = [f"{c['idx']}. {c['description']}" for c in constraints if not c["is_following"]]

        prompt = f"""
You are a revise assistant. Keep all already satisfied constraints intact and fix the unmet ones. Avoid adding new facts.

[Instruction]
{instruction}

[Candidate]
Satisfied: {', '.join(satisfied_desc) if satisfied_desc else 'none'}
Unmet: {', '.join(violated_desc) if violated_desc else 'none'}
Draft:
{candidate.get("response", "")}

Return only the revised answer that satisfies the unmet constraints while preserving the satisfied ones."""
        return prompt.strip()

    def _collect_rewrite_jobs(
        self,
        batch: DataProto,
        prompt_texts: list[str],
        response_texts: list[str],
        reward_infos: list[dict],
        token_rewards: list[float],
        rewrite_cfg: dict[str, Any],
    ) -> tuple[list[RewriteJobSpec], dict[str, float]]:
        threshold = rewrite_cfg.get("threshold", 1.0)
        candidate_limit = rewrite_cfg.get("candidate_limit", 16)
        debug_print = bool(rewrite_cfg.get("debug_print", False))
        debug_print_n = int(rewrite_cfg.get("debug_print_n", 3))
        debug_prompt_chars = int(rewrite_cfg.get("debug_prompt_chars", 400))
        debug_printed = 0
        prompt_mode_raw = rewrite_cfg.get("prompt_mode", rewrite_cfg.get("prompt_style", "constraint"))
        prompt_mode = str(prompt_mode_raw).lower() if prompt_mode_raw is not None else "constraint"
        if prompt_mode in ("constraints", "constraint", "if"):
            prompt_mode = "constraint"
        elif prompt_mode in ("rubrics", "rubric", "medical"):
            prompt_mode = "rubric"
        elif prompt_mode in ("auto", "automatic"):
            prompt_mode = "auto"
        else:
            prompt_mode = "constraint"
        if candidate_limit <= 0:
            return [], {}

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_to_indices[str(uid)].append(idx)

        suffix_counter: dict[str, int] = defaultdict(int)
        rewrite_jobs: list[RewriteJobSpec] = []
        best_score_by_uid: dict[str, float] = {}

        data_source_arr = batch.non_tensor_batch.get("data_source")
        reward_model_arr = batch.non_tensor_batch.get("reward_model")
        extra_info_arr = batch.non_tensor_batch.get("extra_info")
        reward_scores_arr = None
        if reward_infos and isinstance(reward_infos[0], dict):
            reward_scores_arr = reward_infos

        def _get_field(arr, idx):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                arr_len = arr.shape[0]
            else:
                arr_len = len(arr)
            if arr_len == len(batch):
                return arr[idx]
            if arr_len == 1:
                return arr[0]
            return None

        for uid, indices in uid_to_indices.items():
            score_lookup = {i: get_numeric_score(reward_infos[i], token_rewards[i]) for i in indices}
            if not score_lookup:
                continue

            best_score = max(score_lookup.values())
            best_score_by_uid[uid] = best_score

            if best_score >= threshold:
                continue

            best_indices = [
                idx for idx, val in score_lookup.items() if abs(val - best_score) < 1e-6
            ]
            best_indices.sort(
                key=lambda x: sum(
                    normalize_is_following_list(reward_infos[x].get("is_following_list")) or []
                ),
                reverse=True,
            )
            best_indices = best_indices[:candidate_limit]

            if not best_indices:
                continue

            for ci in best_indices:
                selected_mode = prompt_mode
                rubrics: list[str] = []
                constraints: list[dict[str, Any]] = []
                constraint_results = None
                if prompt_mode in ("rubric", "auto"):
                    extra_info = _get_field(extra_info_arr, ci)
                    rubrics = self._extract_rubrics_from_extra_info(extra_info)
                    score_info = reward_infos[ci] if reward_infos and ci < len(reward_infos) else {}
                    if isinstance(score_info, dict):
                        constraint_results = score_info.get("constraint_results")
                    if prompt_mode == "auto" and not rubrics:
                        selected_mode = "constraint"

                if selected_mode == "constraint":
                    constraints = self._extract_constraints_for_prompt(
                        reward_infos[ci],
                        fallback_is_list=normalize_is_following_list(
                            reward_infos[ci].get("is_following_list")
                        ),
                    )
                    candidate = {
                        "response": response_texts[ci],
                        "score": score_lookup[ci],
                        "constraints": constraints,
                    }
                else:
                    candidate = {
                        "response": response_texts[ci],
                        "score": score_lookup[ci],
                        "rubrics": rubrics,
                        "constraint_results": constraint_results,
                    }
                rewrite_prompt = self._build_self_rewrite_prompt(
                    instruction=prompt_texts[indices[0]],
                    candidate=candidate,
                    target_score=threshold,
                    prompt_mode=selected_mode,
                )
                if debug_print and debug_printed < debug_print_n:
                    debug_printed += 1
                    if selected_mode == "rubric":
                        total_rubrics = len(rubrics)
                        if isinstance(constraint_results, list):
                            satisfied = sum(1 for item in constraint_results if item.get("is_following") is True)
                            unmet = sum(1 for item in constraint_results if item.get("is_following") is False)
                            unclear = len(constraint_results) - satisfied - unmet
                            status = f"satisfied={satisfied} unmet={unmet} unclear={unclear}"
                        else:
                            status = "constraint_results=missing"
                        detail = f"rubrics={total_rubrics} {status}"
                    else:
                        satisfied = sum(1 for item in constraints if item.get("is_following") is True)
                        unmet = sum(1 for item in constraints if item.get("is_following") is False)
                        detail = f"constraints={len(constraints)} satisfied={satisfied} unmet={unmet}"
                    print(
                        "[REWRITE][DEBUG] "
                        f"uid={uid} cand_idx={ci} best={best_score:.4f} score={score_lookup[ci]:.4f} "
                        f"{detail}"
                    )
                    if debug_prompt_chars > 0:
                        preview = rewrite_prompt[:debug_prompt_chars]
                        print("[REWRITE][DEBUG] prompt_preview:\n", preview)
                rewrite_jobs.append(
                    RewriteJobSpec(
                        uid=uid,
                        template_idx=indices[0],
                        candidate_idx=ci,
                        rewrite_prompt=rewrite_prompt,
                        best_original_score=best_score,
                        data_source=_get_field(data_source_arr, indices[0]),
                        reward_model=deepcopy(_get_field(reward_model_arr, indices[0])),
                        extra_info=deepcopy(_get_field(extra_info_arr, indices[0])),
                        rollout_reward_scores=deepcopy(_get_field(reward_scores_arr, indices[0])),
                        rewrite_suffix=suffix_counter[uid],
                    )
                )
                suffix_counter[uid] += 1

        return rewrite_jobs, best_score_by_uid

    def _build_rewrite_generation_batch(
        self,
        rewrite_jobs: list[RewriteJobSpec],
        batch: DataProto,
        prompt_length: Optional[int],
        rewrite_cfg: dict[str, Any],
    ) -> DataProto:
        rewrite_prompts = [job.rewrite_prompt for job in rewrite_jobs]
        if rewrite_cfg.get("use_chat_template", False):
            apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
            extra_kwargs = rewrite_cfg.get("apply_chat_template_kwargs") or {}
            if extra_kwargs:
                apply_kwargs.update(extra_kwargs)
            if apply_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "rewrite_he.use_chat_template=True requires tokenizer.chat_template or "
                    "rewrite_he.apply_chat_template_kwargs.chat_template"
                )
            rewrite_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_kwargs,
                )
                for prompt in rewrite_prompts
            ]

        padding_side = getattr(self.tokenizer, "padding_side", None)
        if padding_side is not None:
            self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                rewrite_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
        finally:
            if padding_side is not None:
                self.tokenizer.padding_side = padding_side

        target_length = prompt_length
        if target_length is None or target_length <= 0:
            target_length = encoded["input_ids"].shape[1]
        input_ids, attention_mask = self._pad_or_trim_prompts(
            encoded["input_ids"],
            encoded["attention_mask"],
            target_length,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        non_tensor_batch: dict[str, np.ndarray] = {}
        for key, arr in batch.non_tensor_batch.items():
            values = []
            for job in rewrite_jobs:
                value = arr[job.template_idx]
                if key == "uid":
                    value = f"{value}#rewrite{job.rewrite_suffix}"
                else:
                    value = deepcopy(value)
                values.append(value)
            non_tensor_batch[key] = np.array(values, dtype=object)

        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": rewrite_cfg.get("do_sample", True),
            "temperature": rewrite_cfg.get(
                "temperature",
                self.config.actor_rollout_ref.rollout.temperature,
            ),
            "top_p": rewrite_cfg.get(
                "top_p",
                self.config.actor_rollout_ref.rollout.get("top_p", 1.0),
            ),
            "top_k": rewrite_cfg.get(
                "top_k",
                self.config.actor_rollout_ref.rollout.get("top_k", 0),
            ),
            "response_length": rewrite_cfg.get(
                "response_length",
                self.config.actor_rollout_ref.rollout.response_length,
            ),
        }

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors=non_tensor_batch,
            meta_info=meta_info,
        )

    def _encode_rewrite_prompt_batch(
        self,
        rewrite_prompts: list[str],
        prompt_length: int,
        rewrite_cfg: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not rewrite_prompts:
            return None, None

        if rewrite_cfg.get("use_chat_template", False):
            apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
            extra_kwargs = rewrite_cfg.get("apply_chat_template_kwargs") or {}
            if extra_kwargs:
                apply_kwargs.update(extra_kwargs)
            if apply_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "rewrite_he.use_chat_template=True requires tokenizer.chat_template or "
                    "rewrite_he.apply_chat_template_kwargs.chat_template"
                )
            rewrite_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    **apply_kwargs,
                )
                for prompt in rewrite_prompts
            ]

        padding_side = getattr(self.tokenizer, "padding_side", None)
        if padding_side is not None:
            self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                rewrite_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
        finally:
            if padding_side is not None:
                self.tokenizer.padding_side = padding_side

        target_length = prompt_length
        if target_length is None or target_length <= 0:
            target_length = encoded["input_ids"].shape[1]
        input_ids, attention_mask = self._pad_or_trim_prompts(
            encoded["input_ids"],
            encoded["attention_mask"],
            target_length,
        )
        return input_ids, attention_mask

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    
                    print("batch batch keys:", list(batch.batch.keys()))
                    print("batch non_tensor keys:", list(batch.non_tensor_batch.keys()))
                    print("batch meta_info keys:", list(batch.meta_info.keys()))  

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                            
                        def _update_dense_metric(reward_extra_infos_local: dict[str, list] | None):
                            if not reward_extra_infos_local or "dense" not in reward_extra_infos_local:
                                return
                            dense_vals = reward_extra_infos_local.get("dense")
                            if dense_vals is None:
                                return
                            if isinstance(dense_vals, np.ndarray):
                                dense_iter = dense_vals.tolist()
                            else:
                                dense_iter = dense_vals
                            try:
                                not_one = sum(
                                    1
                                    for v in dense_iter
                                    if v is not None and float(v) != 1.0
                                )
                            except Exception:
                                return
                            metrics["rollout/dense_not_one_count"] = float(not_one)
                            
                            
                        def _update_uid_max_not_one_metric(reward_tensor_local: torch.Tensor, batch_local: DataProto):
                            if reward_tensor_local is None or batch_local is None:
                                return
                            try:
                                reward_list = reward_tensor_local.sum(dim=-1).detach().cpu().tolist()
                            except Exception:
                                return
                            uid_arr = batch_local.non_tensor_batch.get("uid")
                            if uid_arr is None:
                                return
                            uid_iter = uid_arr.tolist() if isinstance(uid_arr, np.ndarray) else uid_arr
                            uid_to_rewards: dict[str, list[float]] = defaultdict(list)
                            for r, uid in zip(reward_list, uid_iter):
                                uid_to_rewards[str(uid)].append(r)
                            not_one = sum(
                                1
                                for vals in uid_to_rewards.values()
                                if vals and max(vals) != 1.0
                            )
                            metrics["rollout/uid_max_reward_not_one_count"] = float(not_one)



                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn) 
                            print("batch batch keys:", list(batch.batch.keys()))
                            print("batch non_tensor keys:", list(batch.non_tensor_batch.keys()))
                            print("batch meta_info keys:", list(batch.meta_info.keys()))   
                            
                            (
                            batch,
                            reward_tensor,
                            reward_extra_infos_dict,
                            rewrite_kept,
                            rewrite_attempted,
                            rewrite_metrics,
                        ) = self._augment_with_rewrite_he(batch, reward_tensor, reward_extra_infos_dict)
                        
                        print("batch length after rewrite:", len(batch))
                        print("batch batch keys:", list(batch.batch.keys()))
                        print("batch non_tensor keys:", list(batch.non_tensor_batch.keys()))
                        print("batch meta_info keys:", list(batch.meta_info.keys()))
                        metrics.update(rewrite_metrics)
                        logger.log(data=metrics, step=self.global_steps) 
                        #progress_bar.close()                 
                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            use_he_logprob = self.config.actor_rollout_ref.actor.policy_loss.get(
                                "use_he_logprob", True
                            )
                            if use_he_logprob:
                                if "he_mask" in batch.batch and "he_log_probs" not in batch.batch:
                                    batch.batch["he_log_probs"] = batch.batch["old_log_probs"]
                            elif "he_log_probs" in batch.batch:
                                batch.batch.pop("he_log_probs")
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        _update_dense_metric(reward_extra_infos_dict)
                        _update_uid_max_not_one_metric(reward_tensor, batch)
                        
                        
                        if reward_extra_infos_dict:
                            reward_extra_infos_dict.pop("is_following_list", None)
                            reward_extra_infos_dict.pop("constraint_results", None)
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
                    
                    
    def _generate_rewrite_with_external_vllm(
        self,
        rewrite_jobs: list[RewriteJobSpec],
        rewrite_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not rewrite_jobs:
            return []

        base_url = rewrite_cfg.get("vllm_url") or os.getenv("REWRITE_VLLM_URL")
        model = (
            rewrite_cfg.get("vllm_model")
            or os.getenv("REWRITE_VLLM_MODEL")
            or self.config.actor_rollout_ref.model.path
        )
        api_key = rewrite_cfg.get("vllm_api_key") or os.getenv("REWRITE_VLLM_API_KEY", "EMPTY")
        endpoint = rewrite_cfg.get("vllm_endpoint") or "/v1/chat/completions"
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        timeout = rewrite_cfg.get("request_timeout", 120)
        num_samples = max(1, int(rewrite_cfg.get("num_samples", rewrite_cfg.get("rewrite_n", 1) or 1)))
        temperature = rewrite_cfg.get("temperature", 0.3)
        max_tokens = rewrite_cfg.get(
            "response_length",
            self.config.actor_rollout_ref.rollout.response_length,
        )

        if not base_url:
            print("[REWRITE] No vLLM endpoint configured, skip rewrite.")
            return []

        url = base_url.rstrip("/") + endpoint
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        results: list[dict[str, Any]] = []
        for job in rewrite_jobs:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": job.rewrite_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": num_samples,
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    print(f"[REWRITE] vLLM request failed ({resp.status_code}): {resp.text[:200]}")
                    continue
                data = resp.json()
                for choice_idx, choice in enumerate(data.get("choices", [])):
                    message = choice.get("message") or {}
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        results.append(
                            {
                                "job": job,
                                "response": content.strip(),
                                "sample_idx": choice_idx,
                            }
                        )
            except Exception as e:
                print(f"[REWRITE] vLLM request error: {e}")
        return results

    def _generate_rewrite_with_rollout(
        self,
        rewrite_jobs: list[RewriteJobSpec],
        batch: DataProto,
        rewrite_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not rewrite_jobs:
            return []

        debug_print = bool(rewrite_cfg.get("debug_print", False))
        debug_print_n = int(rewrite_cfg.get("debug_print_n", 3))

        prompt_length = rewrite_cfg.get("prompt_length")
        rewrite_inputs = self._build_rewrite_generation_batch(
            rewrite_jobs,
            batch,
            prompt_length,
            rewrite_cfg,
        )
        if rewrite_inputs is None or len(rewrite_inputs) == 0:
            return []

        num_samples = max(1, int(rewrite_cfg.get("num_samples", rewrite_cfg.get("rewrite_n", 1) or 1)))

        if debug_print and debug_print_n > 0:
            shown = min(debug_print_n, len(rewrite_jobs))
            print(f"[REWRITE] rollout prompt preview: {shown}/{len(rewrite_jobs)}")
            for i, job in enumerate(rewrite_jobs[:shown]):
                print(f"[REWRITE][PROMPT][{i}] uid={job.uid} template_idx={job.template_idx}")
                print(job.rewrite_prompt)

        if num_samples > 1:
            rewrite_inputs = rewrite_inputs.repeat(repeat_times=num_samples, interleave=True)

        rewrite_inputs.meta_info["global_steps"] = self.global_steps

        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        rewrite_inputs_padded, pad_size = pad_dataproto_to_divisor(rewrite_inputs, size_divisor)
        if not self.async_rollout_mode:
            rewrite_outputs_padded = self.actor_rollout_wg.generate_sequences(rewrite_inputs_padded)
        else:
            rewrite_outputs_padded = self.async_rollout_manager.generate_sequences(rewrite_inputs_padded)
        rewrite_outputs = unpad_dataproto(rewrite_outputs_padded, pad_size=pad_size)

        results: list[dict[str, Any]] = []
        base_jobs = len(rewrite_jobs)
        expected = base_jobs * num_samples
        if len(rewrite_outputs) < expected:
            print(f"[REWRITE] rollout outputs shorter than expected: {len(rewrite_outputs)} < {expected}")

        max_items = min(len(rewrite_outputs), expected)
        for i in range(max_items):
            job_idx = i // num_samples
            if job_idx >= base_jobs:
                break
            job = rewrite_jobs[job_idx]
            rewrite_item = rewrite_outputs[i]
            responses = rewrite_item.batch["responses"]
            resp_len = responses.shape[0]

            valid_response_len = resp_len
            if "response_mask" in rewrite_item.batch.keys():
                response_mask = rewrite_item.batch["response_mask"]
                valid_response_len = int(response_mask.sum().item())
            elif "attention_mask" in rewrite_item.batch.keys():
                attention_mask = rewrite_item.batch["attention_mask"]
                response_tail_mask = attention_mask[-resp_len:]
                valid_response_len = int(response_tail_mask.sum().item())

            valid_response_len = max(0, min(valid_response_len, resp_len))
            response_tokens = responses[:valid_response_len]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            if not isinstance(response_text, str) or not response_text.strip():
                continue

            if debug_print and debug_print_n > 0 and job_idx < debug_print_n:
                print(
                    f"[REWRITE][OUTPUT][{job_idx}][{i % num_samples}] uid={job.uid} template_idx={job.template_idx}"
                )
                print(response_text.strip())

            results.append(
                {
                    "job": job,
                    "response": response_text.strip(),
                    "sample_idx": i % num_samples,
                }
            )
        return results

    def _build_rewrite_batch_from_results(
        self,
        rewrite_results: list[dict[str, Any]],
        batch: DataProto,
    ) -> tuple[Optional[DataProto], list[dict[str, Any]]]:
        if not rewrite_results:
            return None, []

        prompt_len = batch.batch["prompts"].shape[1]
        resp_len = batch.batch["responses"].shape[1]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        rewritten_prompts = []
        rewritten_responses = []
        rewritten_input_ids = []
        rewritten_attention_masks = []
        rewritten_position_ids = []
        rewritten_non_tensors: dict[str, list] = {k: [] for k in batch.non_tensor_batch.keys()}
        meta_list: list[dict[str, Any]] = []

        for item in rewrite_results:
            job: RewriteJobSpec = item["job"]
            response_text: str = item.get("response", "")
            template_idx = job.template_idx

            orig_prompts = batch.batch["prompts"][template_idx]
            orig_input_ids = batch.batch["input_ids"][template_idx]
            orig_attn_mask = batch.batch["attention_mask"][template_idx]
            device = orig_input_ids.device

            encoded = self.tokenizer(
                response_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            rewrite_ids = encoded["input_ids"][0].to(device)

            valid_resp_len = rewrite_ids.size(0)
            rewrite_ids = rewrite_ids[:resp_len]
            if rewrite_ids.size(0) < resp_len:
                pad_len = resp_len - rewrite_ids.size(0)
                rewrite_ids = torch.cat(
                    [rewrite_ids, torch.full((pad_len,), pad_id, dtype=rewrite_ids.dtype, device=device)],
                    dim=0,
                )

            new_input_ids = torch.full_like(orig_input_ids, pad_id)
            new_input_ids[:prompt_len] = orig_prompts
            new_input_ids[prompt_len: prompt_len + resp_len] = rewrite_ids

            new_attn_mask = torch.zeros_like(orig_attn_mask, device=device)
            new_attn_mask[:prompt_len] = orig_attn_mask[:prompt_len]
            valid_resp_len = max(1, min(valid_resp_len, resp_len))
            new_attn_mask[prompt_len: prompt_len + valid_resp_len] = 1

            new_pos_ids = compute_position_id_with_mask(new_attn_mask.unsqueeze(0))[0]

            rewritten_prompts.append(orig_prompts.unsqueeze(0))
            rewritten_responses.append(rewrite_ids.unsqueeze(0))
            rewritten_input_ids.append(new_input_ids.unsqueeze(0))
            rewritten_attention_masks.append(new_attn_mask.unsqueeze(0))
            rewritten_position_ids.append(new_pos_ids.unsqueeze(0))

            for key, arr in batch.non_tensor_batch.items():
                val = arr[template_idx]
                if key == "uid":
                    suffix = f"#rewrite{job.rewrite_suffix}"
                    sample_idx = item.get("sample_idx", 0)
                    if sample_idx:
                        suffix = f"{suffix}_{sample_idx}"
                    val = f"{val}{suffix}"
                else:
                    val = deepcopy(val)
                rewritten_non_tensors[key].append(val)

            meta_list.append(
                {
                    "uid_base": str(job.uid),
                    "response_text": response_text,
                }
            )

        rewrite_batch = DataProto.from_dict(
            tensors={
                "prompts": torch.cat(rewritten_prompts, dim=0),
                "responses": torch.cat(rewritten_responses, dim=0),
                "input_ids": torch.cat(rewritten_input_ids, dim=0),
                "attention_mask": torch.cat(rewritten_attention_masks, dim=0),
                "position_ids": torch.cat(rewritten_position_ids, dim=0),
            },
            non_tensors={k: np.array(v, dtype=object) for k, v in rewritten_non_tensors.items()},
            meta_info={
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "global_steps": self.global_steps,
            },
        )
        rewrite_batch.batch["response_mask"] = compute_response_mask(rewrite_batch)
        return rewrite_batch, meta_list

    def _augment_with_rewrite_he(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos: dict[str, list] | None,
    ) -> tuple[DataProto, torch.Tensor, dict[str, list] | None, int, int, dict[str, float]]:
        rewrite_cfg = self.config.trainer.get(
            "rewrite_he",
            self.config.trainer.get("fuse_he", {}),
        )
        metrics: dict[str, float] = {
            "rewrite/prompts_attempted": 0.0,
            "rewrite/prompts_success": 0.0,
            "rewrite/generated": 0.0,
            "rewrite/kept_samples": 0.0,
            "rewrite/success_rate": 0.0,
            "rewrite/over_prev": 0.0,
            "rewrite/prompt_len_gt_2048": 0.0,
            "rewrite/prompt_len_gt_3072": 0.0,
            "rewrite/prompt_len_gt_4096": 0.0,
        }

        if not rewrite_cfg or not rewrite_cfg.get("enable", False):
            return batch, reward_tensor, reward_extra_infos, 0, 0, metrics

        orig_size = len(batch)
        prompt_texts, response_texts = self._decode_prompts_and_responses(batch)
        reward_infos = self._materialize_reward_infos(reward_extra_infos, orig_size)
        token_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()

        rewrite_jobs, best_orig_by_uid = self._collect_rewrite_jobs(
            batch,
            prompt_texts,
            response_texts,
            reward_infos,
            token_rewards,
            rewrite_cfg,
        )
        rewrite_attempted = len(rewrite_jobs)
        metrics["rewrite/prompts_attempted"] = float(rewrite_attempted)

        if not rewrite_jobs:
            return batch, reward_tensor, reward_extra_infos, 0, rewrite_attempted, metrics

        log_prompt_len_metrics = bool(rewrite_cfg.get("log_prompt_len_metrics", False))
        if log_prompt_len_metrics:
            rewrite_prompts = [job.rewrite_prompt for job in rewrite_jobs]
            _, rewrite_prompt_mask = self._encode_rewrite_prompt_batch(
                rewrite_prompts,
                0,
                rewrite_cfg,
            )
            if rewrite_prompt_mask is not None:
                prompt_lens = rewrite_prompt_mask.sum(dim=-1).detach().cpu().tolist()
                metrics["rewrite/prompt_len_gt_2048"] = float(sum(length > 2048 for length in prompt_lens))
                metrics["rewrite/prompt_len_gt_3072"] = float(sum(length > 3072 for length in prompt_lens))
                metrics["rewrite/prompt_len_gt_4096"] = float(sum(length > 4096 for length in prompt_lens))

        generator = rewrite_cfg.get("generator")
        if generator is None:
            generator = "vllm" if rewrite_cfg.get("vllm_url") else "rollout"
        if generator == "vllm":
            rewrite_results = self._generate_rewrite_with_external_vllm(rewrite_jobs, rewrite_cfg)
        else:
            if generator != "rollout":
                print(f"[REWRITE] Unknown generator '{generator}', falling back to rollout.")
            rewrite_results = self._generate_rewrite_with_rollout(rewrite_jobs, batch, rewrite_cfg)
        metrics["rewrite/generated"] = float(len(rewrite_results))

        if not rewrite_results:
            return batch, reward_tensor, reward_extra_infos, 0, rewrite_attempted, metrics

        rewrite_batch, rewrite_meta = self._build_rewrite_batch_from_results(rewrite_results, batch)
        if rewrite_batch is None or len(rewrite_batch) == 0:
            return batch, reward_tensor, reward_extra_infos, 0, rewrite_attempted, metrics


        rewrite_reward_tensor, rewrite_reward_extra_infos = compute_reward(rewrite_batch, self.reward_fn)
        rewrite_reward_infos = self._materialize_reward_infos(rewrite_reward_extra_infos, len(rewrite_batch))
        token_rewards_rw = rewrite_reward_tensor.sum(dim=-1).detach().cpu().tolist()

        reward_shaping_enable = bool(rewrite_cfg.get("reward_shaping_enable", False))
        reward_shaping_alpha = float(rewrite_cfg.get("reward_shaping_alpha", 0.0) or 0.0)
        if reward_shaping_enable and reward_shaping_alpha > 0.0:
            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)
            response_mask = batch.batch["response_mask"]
            best_rewrite_by_idx: dict[int, float] = {}
            for i, item in enumerate(rewrite_results):
                job = item.get("job")
                cand_idx = getattr(job, "candidate_idx", None) if job is not None else None
                if cand_idx is None:
                    continue
                cand_idx = int(cand_idx)
                rw_score = float(token_rewards_rw[i])
                prev = best_rewrite_by_idx.get(cand_idx)
                if prev is None or rw_score > prev:
                    best_rewrite_by_idx[cand_idx] = rw_score
            shaped = 0
            for cand_idx, rw_score in best_rewrite_by_idx.items():
                if cand_idx < 0 or cand_idx >= orig_size:
                    continue
                orig_score = float(token_rewards[cand_idx])
                new_score = orig_score + reward_shaping_alpha * (rw_score - orig_score)
                delta = new_score - orig_score
                if abs(delta) < 1e-12:
                    continue
                mask = response_mask[cand_idx].to(reward_tensor.device)
                token_count = int(mask.sum().item())
                if token_count <= 0:
                    continue
                delta_per_token = reward_tensor.new_tensor(delta / float(token_count))
                reward_tensor[cand_idx] = reward_tensor[cand_idx] + delta_per_token * mask
                token_rewards[cand_idx] = new_score
                shaped += 1
            if shaped > 0:
                metrics["rewrite/reward_shaping_samples"] = float(shaped)

        rewrite_items_info: list[dict] = []
        uid_arr = rewrite_batch.non_tensor_batch.get("uid", [])
        for i in range(len(rewrite_batch)):
            raw_uid = uid_arr[i] if isinstance(uid_arr, (list, np.ndarray)) else None
            if isinstance(raw_uid, np.ndarray):
                raw_uid = raw_uid.item()
            raw_uid_str = str(raw_uid) if raw_uid is not None else ""
            uid_base = raw_uid_str.split("#rewrite", 1)[0]
            rewrite_info = rewrite_reward_infos[i] if rewrite_reward_infos else {}
            numeric = get_numeric_score(rewrite_info, token_rewards_rw[i])
            dense = float(rewrite_info.get("dense", numeric)) if isinstance(rewrite_info, dict) else float(numeric)
            response_text = rewrite_meta[i].get("response_text") if i < len(rewrite_meta) else ""
            rewrite_items_info.append(
                {
                    "index": i,
                    "raw_uid": raw_uid_str,
                    "uid_base": uid_base,
                    "response": response_text,
                    "score_dict": rewrite_info,
                    "dense": dense,
                    "numeric": numeric,
                }
            )

        rewrites_by_uid: dict[str, list[dict]] = defaultdict(list)
        for item in rewrite_items_info:
            rewrites_by_uid[item["uid_base"]].append(item)

        replace_top_n = int(rewrite_cfg.get("replace_top_n", 2) or 2)
        if replace_top_n < 1:
            replace_top_n = 1

        selected_rewrites_by_uid: dict[str, list[dict]] = {}
        over_prev_count = 0
        for uid_base, items in rewrites_by_uid.items():
            if not items:
                selected_rewrites_by_uid[uid_base] = []
                continue
            orig_best = best_orig_by_uid.get(uid_base, float("-inf"))
            sorted_items = sorted(items, key=lambda x: x["numeric"], reverse=True)
            selected: list[dict] = []
            selected_indices: set[int] = set()
            for item in sorted_items:
                if item["numeric"] > orig_best:
                    selected.append(item)
                    selected_indices.add(int(item["index"]))
                    if len(selected) >= replace_top_n:
                        break
            if selected:
                over_prev_count += 1
            if len(selected) < replace_top_n:
                for item in sorted_items:
                    if item["numeric"] >= orig_best and int(item["index"]) not in selected_indices:
                        selected.append(item)
                        selected_indices.add(int(item["index"]))
                        if len(selected) >= replace_top_n:
                            break
            selected_rewrites_by_uid[uid_base] = selected

        metrics["rewrite/over_prev"] = float(over_prev_count)

        uid_to_indices2: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_to_indices2[str(uid)].append(idx)

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        best_orig_items_by_uid: dict[str, list[dict[str, Any]]] = {}
        for uid_base, indices in uid_to_indices2.items():
            scored_items: list[dict[str, Any]] = []
            for idx in indices:
                numeric = get_numeric_score(reward_infos[idx], token_rewards[idx])
                follow = sum(
                    normalize_is_following_list(reward_infos[idx].get("is_following_list")) or []
                )
                scored_items.append(
                    {
                        "orig_idx": int(idx),
                        "numeric": float(numeric),
                        "follow": float(follow),
                    }
                )
            scored_items.sort(key=lambda x: (x["numeric"], x["follow"]), reverse=True)
            if scored_items:
                best_orig_items_by_uid[uid_base] = scored_items

        if reward_extra_infos is not None:
            for k, v in reward_extra_infos.items():
                if isinstance(v, np.ndarray):
                    reward_extra_infos[k] = v.tolist()

        prompt_len = batch.batch["prompts"].shape[1]
        resp_len = batch.batch["responses"].shape[1]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        train_use_rewrite_prompt = bool(rewrite_cfg.get("train_use_rewrite_prompt", False))
        use_rewrite_prompt = train_use_rewrite_prompt
        rewrite_prompt_ids = None
        rewrite_prompt_mask = None
        if use_rewrite_prompt:
            prompt_texts_for_rewrite = [item["job"].rewrite_prompt for item in rewrite_results]
            rewrite_prompt_ids, rewrite_prompt_mask = self._encode_rewrite_prompt_batch(
                prompt_texts_for_rewrite,
                prompt_len,
                rewrite_cfg,
            )
            if rewrite_prompt_ids is None or rewrite_prompt_mask is None:
                use_rewrite_prompt = False

        attempted_uids = {str(job.uid) for job in rewrite_jobs}
        selected_by_uid: dict[str, list[dict[str, Any]]] = {}
        for uid_base in attempted_uids:
            selected_items: list[dict[str, Any]] = []
            for item in selected_rewrites_by_uid.get(uid_base, []):
                selected_items.append(
                    {
                        "source": "rewrite",
                        "uid_base": uid_base,
                        "rewrite_idx": int(item["index"]),
                        "numeric": float(item["numeric"]),
                    }
                )
            if len(selected_items) < replace_top_n:
                for orig_item in best_orig_items_by_uid.get(uid_base, []):
                    selected_items.append(
                        {
                            "source": "orig",
                            "uid_base": uid_base,
                            "orig_idx": int(orig_item["orig_idx"]),
                            "numeric": float(orig_item["numeric"]),
                        }
                    )
                    if len(selected_items) >= replace_top_n:
                        break
            if selected_items:
                selected_by_uid[uid_base] = selected_items

        if not selected_by_uid:
            return batch, reward_tensor, reward_extra_infos, 0, rewrite_attempted, metrics

        mask_mode = str(rewrite_cfg.get("response_mask_mode", "length")).lower()
        use_eos_mask = mask_mode in ("eos", "rollout")
        append_eos = bool(rewrite_cfg.get("append_eos", True))
        compare_mask = bool(rewrite_cfg.get("compare_response_mask", False))
        diff_samples = 0
        diff_tokens = 0
        compared = 0
        eos_token_id = self.tokenizer.eos_token_id if (compare_mask or use_eos_mask) else None

        rng = np.random.default_rng()
        replaced = 0
        rewrite_replaced = 0
        fallback_replaced = 0
        rewrite_indices: list[int] = []

        rewrite_uid_bases: set[str] = set()
        for uid_base, infos in selected_by_uid.items():
            indices = uid_to_indices2.get(uid_base)
            if not indices:
                continue
            indices = list(indices)
            rng.shuffle(indices)
            if len(infos) > len(indices):
                infos = sorted(
                    infos,
                    key=lambda x: (
                        1 if x.get("source") == "rewrite" else 0,
                        x.get("numeric", float("-inf")),
                    ),
                    reverse=True,
                )
                infos = infos[: len(indices)]

            for slot_idx, info in enumerate(infos):
                replace_idx = indices[slot_idx]
                orig_prompts = batch.batch["prompts"][replace_idx]
                orig_input_ids = batch.batch["input_ids"][replace_idx]
                orig_attn_mask = batch.batch["attention_mask"][replace_idx]
                device = orig_input_ids.device
                prompt_ids = orig_prompts
                prompt_mask = orig_attn_mask[:prompt_len]

                if info["source"] == "rewrite":
                    rewrite_idx = int(info["rewrite_idx"])
                    response_ids = rewrite_batch.batch["responses"][rewrite_idx].to(device)
                    response_mask = rewrite_batch.batch["response_mask"][rewrite_idx].to(device)
                    orig_response_mask = response_mask
                    if use_rewrite_prompt and rewrite_prompt_ids is not None:
                        prompt_ids = rewrite_prompt_ids[rewrite_idx].to(device)
                        prompt_mask = rewrite_prompt_mask[rewrite_idx].to(
                            device=device, dtype=orig_attn_mask.dtype
                        )

                    if use_eos_mask and eos_token_id is not None:
                        response_ids = response_ids.clone()
                        valid_len = int(orig_response_mask.sum().item())
                        if append_eos and valid_len < resp_len:
                            response_ids[valid_len] = eos_token_id
                        response_mask = get_response_mask(
                            response_ids.unsqueeze(0),
                            eos_token=eos_token_id,
                            dtype=orig_response_mask.dtype,
                        )[0]

                    if compare_mask and eos_token_id is not None:
                        eos_mask = get_response_mask(
                            response_ids.unsqueeze(0),
                            eos_token=eos_token_id,
                            dtype=orig_response_mask.dtype,
                        )[0]
                        diff = int((orig_response_mask != eos_mask).sum().item())
                        if diff > 0:
                            diff_samples += 1
                            diff_tokens += diff
                        compared += 1
                else:
                    orig_idx = int(info["orig_idx"])
                    response_ids = batch.batch["responses"][orig_idx].to(device)
                    response_mask = batch.batch["response_mask"][orig_idx].to(device)

                new_input_ids = torch.full_like(orig_input_ids, pad_id)
                new_input_ids[:prompt_len] = prompt_ids
                new_input_ids[prompt_len: prompt_len + resp_len] = response_ids

                new_attn_mask = torch.zeros_like(orig_attn_mask, device=device)
                new_attn_mask[:prompt_len] = prompt_mask
                new_attn_mask[prompt_len: prompt_len + resp_len] = response_mask

                new_pos_ids = compute_position_id_with_mask(new_attn_mask.unsqueeze(0))[0]

                batch.batch["prompts"][replace_idx] = prompt_ids
                batch.batch["responses"][replace_idx] = response_ids
                batch.batch["response_mask"][replace_idx] = response_mask
                batch.batch["input_ids"][replace_idx] = new_input_ids
                batch.batch["attention_mask"][replace_idx] = new_attn_mask
                batch.batch["position_ids"][replace_idx] = new_pos_ids

                if info["source"] == "rewrite":
                    rewrite_indices.append(replace_idx)
                    rewrite_uid_bases.add(uid_base)
                    reward_tensor[replace_idx] = rewrite_reward_tensor[rewrite_idx].to(reward_tensor.device)
                    if reward_extra_infos is not None:
                        rewrite_info = rewrite_reward_infos[rewrite_idx] if rewrite_reward_infos else {}
                        if isinstance(rewrite_info, dict):
                            for k in rewrite_info.keys():
                                if k not in reward_extra_infos:
                                    reward_extra_infos[k] = [None] * orig_size
                            for k, v in rewrite_info.items():
                                reward_extra_infos[k][replace_idx] = v
                    rewrite_replaced += 1
                else:
                    reward_tensor[replace_idx] = reward_tensor[orig_idx].to(reward_tensor.device)
                    if reward_extra_infos is not None:
                        for k, values in reward_extra_infos.items():
                            reward_extra_infos[k][replace_idx] = values[orig_idx]
                    fallback_replaced += 1

                replaced += 1

        if replaced == 0:
            return batch, reward_tensor, reward_extra_infos, 0, rewrite_attempted, metrics

        onpolicy_compute = bool(rewrite_cfg.get("train_use_onpolicy_compute", False))
        
        # if rewrite_indices:
        #     if train_use_rewrite_prompt:
        #         # Treat rewrite samples as on-policy when training with rewrite prompts.
        #         batch.batch.pop("he_mask", None)
        #     else:
        he_mask = torch.zeros_like(batch.batch["response_mask"])
        he_mask[rewrite_indices] = batch.batch["response_mask"][rewrite_indices]
        batch.batch["he_mask"] = he_mask

        if compare_mask and compared > 0:
            metrics["rewrite/response_mask_diff_samples"] = float(diff_samples)
            metrics["rewrite/response_mask_diff_tokens"] = float(diff_tokens)
            metrics["rewrite/response_mask_diff_rate"] = float(diff_samples) / float(compared)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        metrics["rewrite/kept_samples"] = float(replaced)
        metrics["rewrite/prompts_success"] = float(len(rewrite_uid_bases))
        metrics["rewrite/success_rate"] = float(len(rewrite_uid_bases)) / max(1.0, float(rewrite_attempted))
        metrics["rewrite/rewrite_samples"] = float(rewrite_replaced)
        metrics["rewrite/fallback_samples"] = float(fallback_replaced)

        return batch, reward_tensor, reward_extra_infos, replaced, rewrite_attempted, metrics
