#!/bin/bash
set -euo pipefail

WAIT_PID="${1:-}"

PROJECT_ROOT=/workspace/SDPO-new-clean
VENV_ROOT=/workspace/SIPO/.venv
RUN_SUFFIX="${RUN_SUFFIX:-}"
QUEUE_LOG="$PROJECT_ROOT/logs/qwen3-4b-tooluse-sdpo-variants-after-${WAIT_PID:-now}-$(date -u +%Y%m%d-%H%M%S).log"

mkdir -p "$PROJECT_ROOT/logs"
echo "$$" > "$PROJECT_ROOT/logs/qwen3-4b-tooluse-sdpo-variants-after.pid"
echo "$QUEUE_LOG" > "$PROJECT_ROOT/logs/qwen3-4b-tooluse-sdpo-variants-after.logpath"

log() {
    echo "[$(date -u +"%F %T UTC")] $*"
}

stop_ray() {
    "$VENV_ROOT/bin/python" -m ray.scripts.scripts stop --force || true
}

wait_for_pid() {
    local pid="$1"
    if [[ -z "$pid" || "$pid" == "now" ]]; then
        return
    fi
    log "Waiting for PID ${pid} before tooluse SDPO variants."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "PID ${pid} finished; starting tooluse SDPO variants."
}

run_tooluse_variant() {
    local exp_name="$1"
    shift

    log "Stopping any existing Ray runtime before ${exp_name}."
    stop_ray
    log "Starting ${exp_name}."

    bash training/verl_training.sh \
        "$exp_name" \
        sdpo \
        datasets/tooluse \
        max_model_len=10240 \
        data.train_files=[/workspace/SIPO/datasets/tooluse/train.parquet] \
        data.val_files=[/workspace/SIPO/datasets/tooluse/test.parquet] \
        data.train_batch_size=32 \
        data.train_max_samples=4800 \
        data.max_prompt_length=2048 \
        data.max_response_length=8192 \
        data.apply_chat_template_kwargs.enable_thinking=false \
        trainer.group_name=QWEN3-SDPO-TR-GRPO-matched-generalization-NEW \
        trainer.n_gpus_per_node=8 \
        trainer.total_training_steps=200 \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.save_best_checkpoint=True \
        trainer.best_checkpoint_metric=val-aux/tooluse/reward/mean@16 \
        trainer.best_checkpoint_mode=max \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.max_critic_ckpt_to_keep=1 \
        trainer.default_local_dir="$PROJECT_ROOT/checkpoints/datasets/tooluse/$exp_name" \
        custom_reward_function.path="$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py" \
        +ray_kwargs.ray_init._temp_dir=/tmp/ray_new_q3g_tooluse_sdpo_Qwen3_4B \
        +ray_kwargs.ray_init.include_dashboard=False \
        actor_rollout_ref.model.path=Qwen/Qwen3-4B \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.01 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.max_model_len=10240 \
        actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.val_kwargs.n=16 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        algorithm.norm_adv_by_std_in_grpo=False \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=2.0 \
        actor_rollout_ref.model.use_fused_kernels=False \
        actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
        actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
        actor_rollout_ref.actor.self_distillation.alpha=0.5 \
        actor_rollout_ref.actor.self_distillation.teacher_regularization=trust-region \
        actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.1 \
        actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
        actor_rollout_ref.actor.self_distillation.jsd_histogram_log_freq=5 \
        actor_rollout_ref.actor.self_distillation.jsd_histogram_max_samples=8192 \
        actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
        actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
        "$@"

    log "Finished ${exp_name}."
}

main() {
    cd "$PROJECT_ROOT"
    source "$VENV_ROOT/bin/activate"

    export USER="${USER:-root}"
    export PYTHONPATH="$PROJECT_ROOT"
    export WORKSPACE_DIR=/workspace/SIPO
    export SKIP_INSTALL=true
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    export NUM_GPUS="${NUM_GPUS:-8}"
    export WANDB_ENTITY="${WANDB_ENTITY:-seongryongjung-chung-ang-university}"

    wait_for_pid "$WAIT_PID"

    run_tooluse_variant \
        qwen3gen-tooluse-SDPO_TR-Qwen-Qwen3-4B-mbs32-tr0.1-train32-rollout8-lr1e-5-vllm0.8-newrepo-sdpo-200clean${RUN_SUFFIX}

    run_tooluse_variant \
        qwen3gen-tooluse-SDPO_TR-Qwen-Qwen3-4B-mbs32-tr0.1-train32-rollout8-lr1e-5-vllm0.8-newrepo-sdpo-cmoe03-200clean${RUN_SUFFIX} \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.3 \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=moe \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.0 \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=moe

    run_tooluse_variant \
        qwen3gen-tooluse-SDPO_TR-Qwen-Qwen3-4B-mbs32-tr0.1-train32-rollout8-lr1e-5-vllm0.8-newrepo-sdpo-cpoe03-200clean${RUN_SUFFIX} \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.3 \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=poe \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.0 \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=poe

    run_tooluse_variant \
        qwen3gen-tooluse-SDPO_TR-Qwen-Qwen3-4B-mbs32-tr0.1-train32-rollout8-lr1e-5-vllm0.8-newrepo-sdpo-imoe03-200clean${RUN_SUFFIX} \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.0 \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=moe \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.3 \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=moe

    run_tooluse_variant \
        qwen3gen-tooluse-SDPO_TR-Qwen-Qwen3-4B-mbs32-tr0.1-train32-rollout8-lr1e-5-vllm0.8-newrepo-sdpo-ipoe03-200clean${RUN_SUFFIX} \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.0 \
        actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=poe \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.3 \
        actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=poe

    log "All tooluse SDPO variants finished."
} >> "$QUEUE_LOG" 2>&1

main
