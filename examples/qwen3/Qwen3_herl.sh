set -x



export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.enable_fuse=True \
    +trainer.rewrite_he.train_use_rewrite_prompt=True \
    +trainer.rewrite_he.gamma=0.1 \
    +trainer.rewrite_he.replace_top_n=1 \
    +trainer.rewrite_he.reward_shaping_enable=True \
    +trainer.rewrite_he.reward_shaping_alpha=0.05 \
    +trainer.rewrite_he.debug_print=False \
    +trainer.rewrite_he.prompt_mode=IF_hir \
    +trainer.rewrite_he.log_prompt_len_metrics=false \
    data.train_files=your_IFdata_path \
    data.val_files=your_IFdata_path \
    data.train_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=your_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +trainer.rewrite_he.enable=True \
    +trainer.rewrite_he.generator=rollout \
    +trainer.rewrite_he.threshold=1.0 \
    +trainer.rewrite_he.candidate_limit=16 \
    +trainer.rewrite_he.temperature=0.3 \
    +trainer.rewrite_he.response_length=2048 \
    +trainer.rewrite_he.do_sample=True \
    +trainer.rewrite_he.compare_response_mask=true \
    +trainer.rewrite_he.response_mask_mode=eos \
    +trainer.rewrite_he.append_eos=true \
    actor_rollout_ref.actor.policy_loss.loss_mode=mixpolicy_vanilla \
    actor_rollout_ref.actor.policy_loss.he_shaping_gamma=1 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.default_local_dir=your_local_path \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='your_name' \
    trainer.experiment_name='your_name' \
    custom_reward_function.path=/yourpath/verl/verl/utils/reward_score/if_score/reward_compute.py \
    custom_reward_function.name=compute_score \
    data.reward_fn_key=data_source \
    reward_model.reward_manager=naive \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_epochs=6 $@
