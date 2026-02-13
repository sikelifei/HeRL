set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ===== placeholders (fill these) =====
MODEL_PATH=""
TRAIN_FILES=""
VAL_FILES=""
LOCAL_DIR=""
PROJECT_NAME=""
EXPERIMENT_NAME=""

MED_REWARD_PATH=""   #path to rarrewardpy

API_BASE_URL=""
API_KEY=""


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +trainer.rewrite_he.prompt_mode=medical_rar \
    +algorithm.enable_fuse=True \
    +trainer.rewrite_he.train_use_rewrite_prompt=True \
    +trainer.rewrite_he.gamma=0.1 \
    +trainer.rewrite_he.replace_top_n=1 \
    +trainer.rewrite_he.reward_shaping_enable=True \
    +trainer.rewrite_he.reward_shaping_alpha=0.05 \
    +trainer.rewrite_he.enable=True \
    +trainer.rewrite_he.generator=rollout \
    +trainer.rewrite_he.threshold=1.0 \
    +trainer.rewrite_he.candidate_limit=16 \
    +trainer.rewrite_he.temperature=0.3 \
    +trainer.rewrite_he.response_length=4096 \
    +trainer.rewrite_he.do_sample=True \
    +trainer.rewrite_he.compare_response_mask=true \
    +trainer.rewrite_he.response_mask_mode=eos \
    +trainer.rewrite_he.append_eos=true \
    actor_rollout_ref.actor.policy_loss.loss_mode=mixpolicy_vanilla \
    actor_rollout_ref.actor.policy_loss.he_shaping_gamma=0.1 \
    trainer.val_before_train=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.default_local_dir="${LOCAL_DIR}" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    custom_reward_function.path="${MED_REWARD_PATH}" \
    custom_reward_function.name=rarcompute_score \
    reward_model.reward_manager=batch \
    +reward_model.reward_kwargs.base_url="${API_BASE_URL}" \
    +reward_model.reward_kwargs.model="gpt-4o-mini" \
    +reward_model.reward_kwargs.api_key="${API_KEY}" \
    +reward_model.reward_kwargs.max_concurrency=512 \
    +reward_model.reward_kwargs.timeout_sec=60 \
    +reward_model.reward_kwargs.max_retries=2 \
    +reward_model.reward_kwargs.retry_backoff_sec=0.5 \
    +reward_model.reward_kwargs.temperature=0.1 \
    +reward_model.reward_kwargs.max_tokens=4096 \
    +reward_model.reward_kwargs.score_scale=1 \
    data.reward_fn_key=data_source \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@