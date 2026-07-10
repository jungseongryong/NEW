# GRPO/SDPO/Mix/SRPO/RLSD/RLRT Reproduction Notes

이 문서는 2026-07-10 기준 `/workspace/SDPO-new-clean`에서 돌리고 있는 Qwen3-4B 실험을 나중에 그대로 재현하기 위한 기록이다. 핵심 목표는 다음 세 가지다.

- 현재 예약된 SDPO+mix / plain SDPO 실험을 어떤 환경과 하이퍼파라미터로 돌렸는지 남긴다.
- 로컬 로그, W&B 로그, best checkpoint, JSD histogram JSONL이 어디에 남는지 남긴다.
- 같은 세팅에서 `grpo`, `sdpo`, `sdpo+mix`, `srpo`, `rlsd`, `rlrt`를 바꿔 돌리는 방법을 남긴다.

## Repository

원격 저장소:

```bash
git clone https://github.com/jungseongryong/NEW.git
cd NEW
```

현재 작업 디렉터리:

```bash
/workspace/SDPO-new-clean
```

주요 재현 스크립트:

```bash
scripts/queue_new_chemistry_sdpo_then_cmoe05_after_pid.sh
scripts/queue_new_tooluse_sdpo_after_pid.sh
scripts/queue_new_all_datasets_plain_sdpo_after_pid.sh
training/verl_training.sh
```

## Python / Environment

현재 머신에서 실제로 사용한 가상환경은 `/workspace/SIPO/.venv`다.

```bash
cd /workspace/SDPO-new-clean
source /workspace/SIPO/.venv/bin/activate

export USER="${USER:-root}"
export PYTHONPATH=/workspace/SDPO-new-clean
export WORKSPACE_DIR=/workspace/SIPO
export SKIP_INSTALL=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export WANDB_ENTITY=seongryongjung-chung-ang-university
```

실제 Python:

```bash
/workspace/SIPO/.venv/bin/python
```

Ray는 각 run 시작 전에 강제로 정리한다.

```bash
/workspace/SIPO/.venv/bin/python -m ray.scripts.scripts stop --force
```

## Data

현재 재현 스크립트는 parquet 데이터를 `/workspace/SIPO/datasets`에서 읽는다.

```text
/workspace/SIPO/datasets/sciknoweval/chemistry/train.parquet
/workspace/SIPO/datasets/sciknoweval/chemistry/test.parquet
/workspace/SIPO/datasets/sciknoweval/physics/train.parquet
/workspace/SIPO/datasets/sciknoweval/physics/test.parquet
/workspace/SIPO/datasets/sciknoweval/biology/train.parquet
/workspace/SIPO/datasets/sciknoweval/biology/test.parquet
/workspace/SIPO/datasets/sciknoweval/material/train.parquet
/workspace/SIPO/datasets/sciknoweval/material/test.parquet
/workspace/SIPO/datasets/tooluse/train.parquet
/workspace/SIPO/datasets/tooluse/test.parquet
```

metric prefix:

```text
sciknoweval datasets: val-aux/sciknoweval/reward/mean@16
tooluse:              val-aux/tooluse/reward/mean@16
```

## Common Hyperparameters

현재 예약된 run들은 아래 공통 세팅을 쓴다.

```text
model: Qwen/Qwen3-4B
n_gpus_per_node: 8
train_batch_size: 32
ppo_mini_batch_size: 32
rollout.n: 8
rollout.temperature: 1.0
rollout.top_p: 1.0
rollout.max_model_len: 10240
rollout.max_num_batched_tokens: 10240
rollout.gpu_memory_utilization: 0.8
rollout.calculate_log_probs: true
validation n: 16
validation temperature: 0.6
validation top_p: 0.95
validation do_sample: true
optim.lr: 1e-5
optim.lr_warmup_steps: 10
optim.weight_decay: 0.01
grad_clip: 1.0
max_prompt_length: 2048
max_response_length: 8192
apply_chat_template_kwargs.enable_thinking: false
norm_adv_by_std_in_grpo: false
rollout_correction.rollout_is: token
rollout_correction.rollout_is_threshold: 2.0
use_fused_kernels: false
```

SDPO/TR self-distillation 공통 세팅:

```text
policy_loss.loss_mode: sdpo
self_distillation.distillation_topk: 100
self_distillation.alpha: 0.5
self_distillation.teacher_regularization: trust-region
self_distillation.teacher_update_rate: 0.1
self_distillation.is_clip: 2.0
self_distillation.include_environment_feedback: false
self_distillation.max_reprompt_len: 10240
self_distillation.jsd_histogram_log_freq: 5
self_distillation.jsd_histogram_max_samples: 8192
```

`teacher_regularization=trust-region`의 의미:

```text
teacher_logits = 0.9 * ref_logits + 0.1 * current_student_logits
```

여기서 interpolation은 probability space가 아니라 logit space에서 일어난다. teacher는 EMA로 업데이트되지 않고 매 forward마다 ref/student logits를 섞어서 만들어진다.

## Saving / Evaluation

현재 예약된 run들은 아래 저장 정책을 쓴다.

```text
trainer.val_before_train: false
trainer.test_freq: 5
trainer.save_freq: -1
trainer.save_best_checkpoint: true
trainer.best_checkpoint_mode: max
trainer.max_actor_ckpt_to_keep: 1
trainer.max_critic_ckpt_to_keep: 1
```

best checkpoint 위치 예시:

```text
/workspace/SDPO-new-clean/checkpoints/datasets/sciknoweval/chemistry/<run_name>/global_step_<N>/actor
/workspace/SDPO-new-clean/checkpoints/datasets/tooluse/<run_name>/global_step_<N>/actor
```

best step tracker:

```text
<checkpoint_dir>/latest_checkpointed_iteration.txt
```

## JSD Histogram Logging

scalar metric은 W&B/console metric pipeline으로 간다. 주요 key:

```text
self_distillation/jsd/token_mean
self_distillation/jsd/correct_token_mean
self_distillation/jsd/incorrect_token_mean
self_distillation/jsd/correct_seq_mean
self_distillation/jsd/incorrect_seq_mean
```

주의: 현재 `token_mean`은 microbatch reduce 방식 때문에 correct/incorrect mean의 단순 중간값처럼 해석하면 안 된다. 분석할 때는 `correct_token_mean`과 `incorrect_token_mean`을 우선 본다.

histogram raw values는 각 run checkpoint directory 아래 JSONL로 저장된다.

```text
<checkpoint_dir>/jsd_histograms.jsonl
```

각 line은 correct/incorrect token 중 하나이며, 예시는 다음과 같다.

```json
{
  "step": 5,
  "key": "self_distillation/jsd_hist/correct_token",
  "count": 41397,
  "mean": 0.021014,
  "std": 0.049853,
  "min": 0.0,
  "p10": 0.0,
  "p25": 0.0,
  "p50": 0.000574,
  "p75": 0.014402,
  "p90": 0.065076,
  "max": 0.342010,
  "values": [...]
}
```

`jsd_histogram_max_samples=8192`는 worker/microbatch 단위에서 적용되고 driver에서 concat되므로, 최종 JSONL의 `count`는 8192보다 클 수 있다. 이 방식은 의도적으로 더 많은 token value를 보존하기 위한 현재 실험 설정이다.

PNG histogram 예시:

```text
docs/figures/jsd_hist_step5_sdpo_cmoe03_jsdhist3.png
```

## Current Queue

현재 예약 구조는 watcher PID를 연결하는 방식이다.

```text
chemistry queue PID -> tooluse variants watcher -> all-dataset plain SDPO watcher
```

2026-07-10에 걸어둔 순서:

```text
1. chemistry sdpo-cmoe03, 150 steps
   correct MoE 0.3, incorrect MoE 0.0

2. chemistry sdpo-cpoe03, 150 steps
   correct PoE 0.3, incorrect PoE 0.0

3. chemistry sdpo-imoe03, 150 steps
   correct MoE 0.0, incorrect MoE 0.3

4. chemistry sdpo-ipoe03, 150 steps
   correct PoE 0.0, incorrect PoE 0.3

5. tooluse plain sdpo, 200 steps

6. tooluse sdpo-cmoe03, 200 steps
   correct MoE 0.3, incorrect MoE 0.0

7. tooluse sdpo-cpoe03, 200 steps
   correct PoE 0.3, incorrect PoE 0.0

8. tooluse sdpo-imoe03, 200 steps
   correct MoE 0.0, incorrect MoE 0.3

9. tooluse sdpo-ipoe03, 200 steps
   correct PoE 0.0, incorrect PoE 0.3

10. chemistry plain sdpo, 200 steps
11. physics plain sdpo, 200 steps
12. biology plain sdpo, 200 steps
13. material plain sdpo, 200 steps
```

tooluse는 마지막 all-dataset plain SDPO queue에서는 제외했다. 이미 tooluse plain/mix variants를 앞에서 200 step씩 돌리기 때문이다.

## Queue Scripts

### Chemistry SDPO/Mix Queue

처음부터 plain SDPO까지 포함해서 실행:

```bash
cd /workspace/SDPO-new-clean
source /workspace/SIPO/.venv/bin/activate

RUN_SUFFIX=-myrun bash scripts/queue_new_chemistry_sdpo_then_cmoe05_after_pid.sh now all
```

plain SDPO를 skip하고 mix variants부터 실행:

```bash
RUN_SUFFIX=-myrun bash scripts/queue_new_chemistry_sdpo_then_cmoe05_after_pid.sh now remaining
```

다른 PID가 끝난 뒤 실행:

```bash
RUN_SUFFIX=-myrun bash scripts/queue_new_chemistry_sdpo_then_cmoe05_after_pid.sh <WAIT_PID> all
```

### Tooluse SDPO/Mix Queue

tooluse plain SDPO + 네 가지 mix variants를 200 step씩 실행한다.

```bash
RUN_SUFFIX=-myrun bash scripts/queue_new_tooluse_sdpo_after_pid.sh now
```

다른 PID 뒤에 붙이기:

```bash
setsid bash -c 'RUN_SUFFIX=-myrun bash scripts/queue_new_tooluse_sdpo_after_pid.sh <WAIT_PID>' \
  >/dev/null 2>&1 &
```

### Plain SDPO On Science Datasets

chemistry/physics/biology/material에 대해 plain SDPO 200 step씩 실행한다. tooluse는 제외되어 있다.

```bash
RUN_SUFFIX=-myrun bash scripts/queue_new_all_datasets_plain_sdpo_after_pid.sh now
```

tooluse queue 뒤에 붙이기:

```bash
setsid bash -c 'RUN_SUFFIX=-aftertooluse bash scripts/queue_new_all_datasets_plain_sdpo_after_pid.sh <TOOLUSE_QUEUE_PID>' \
  >/dev/null 2>&1 &
```

## Run A Single Experiment

아래는 단일 dataset/method를 직접 실행하는 기본 형태다.

```bash
cd /workspace/SDPO-new-clean
source /workspace/SIPO/.venv/bin/activate

export USER="${USER:-root}"
export PYTHONPATH=/workspace/SDPO-new-clean
export WORKSPACE_DIR=/workspace/SIPO
export SKIP_INSTALL=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export WANDB_ENTITY=seongryongjung-chung-ang-university

/workspace/SIPO/.venv/bin/python -m ray.scripts.scripts stop --force
```

### Plain SDPO

```bash
bash training/verl_training.sh \
  qwen3gen-chemistry-SDPO_TR-Qwen-Qwen3-4B-mbs32-tr0.1-train32-rollout8-lr1e-5-vllm0.8-repro-sdpo-200 \
  sdpo \
  datasets/sciknoweval/chemistry \
  max_model_len=10240 \
  data.train_files=[/workspace/SIPO/datasets/sciknoweval/chemistry/train.parquet] \
  data.val_files=[/workspace/SIPO/datasets/sciknoweval/chemistry/test.parquet] \
  data.train_batch_size=32 \
  data.train_max_samples=6400 \
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
  trainer.best_checkpoint_metric=val-aux/sciknoweval/reward/mean@16 \
  trainer.best_checkpoint_mode=max \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=/workspace/SDPO-new-clean/checkpoints/datasets/sciknoweval/chemistry/qwen3gen-chemistry-SDPO_TR-repro-sdpo-200 \
  custom_reward_function.path=/workspace/SDPO-new-clean/verl/utils/reward_score/feedback/__init__.py \
  +ray_kwargs.ray_init._temp_dir=/tmp/ray_new_q3g_chemistry_sdpo_Qwen3_4B \
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
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240
```

### SDPO+Mix

SDPO+mix는 plain SDPO 명령에 아래 override를 추가한다.

correct MoE 0.3 / incorrect MoE 0.0:

```bash
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.3 \
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=moe \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.0 \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=moe
```

correct PoE 0.3 / incorrect PoE 0.0:

```bash
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.3 \
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=poe \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.0 \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=poe
```

correct MoE 0.0 / incorrect MoE 0.3:

```bash
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.0 \
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=moe \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.3 \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=moe
```

correct PoE 0.0 / incorrect PoE 0.3:

```bash
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_student_weight=0.0 \
actor_rollout_ref.actor.self_distillation.sdpo_correct_teacher_mix_mode=poe \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_student_weight=0.3 \
actor_rollout_ref.actor.self_distillation.sdpo_incorrect_teacher_mix_mode=poe
```

MoE target:

```text
target = beta * stopgrad(student_distribution) + (1 - beta) * stopgrad(teacher_distribution)
```

PoE target:

```text
target_logp = normalize(beta * student_logp + (1 - beta) * teacher_logp)
```

top-k SDPO+mix는 full-vocab mix target을 만든 뒤 student top-k index로 gather한다. 즉 top-k selection은 student 기준이고, target value는 mixed teacher target에서 가져온다.

## GRPO

GRPO는 self-distillation/TR을 쓰지 않는 baseline이다. 이 코드베이스에서는 `baseline_grpo` config를 사용한다.

기본 실행 형태:

```bash
bash training/verl_training.sh \
  qwen3gen-chemistry-GRPO-Qwen-Qwen3-4B-repro-200 \
  baseline_grpo \
  datasets/sciknoweval/chemistry \
  max_model_len=10240 \
  data.train_files=[/workspace/SIPO/datasets/sciknoweval/chemistry/train.parquet] \
  data.val_files=[/workspace/SIPO/datasets/sciknoweval/chemistry/test.parquet] \
  data.train_batch_size=32 \
  data.train_max_samples=6400 \
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
  trainer.best_checkpoint_metric=val-aux/sciknoweval/reward/mean@16 \
  trainer.best_checkpoint_mode=max \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=/workspace/SDPO-new-clean/checkpoints/datasets/sciknoweval/chemistry/qwen3gen-chemistry-GRPO-repro-200 \
  custom_reward_function.path=/workspace/SDPO-new-clean/verl/utils/reward_score/feedback/__init__.py \
  +ray_kwargs.ray_init._temp_dir=/tmp/ray_new_q3g_chemistry_grpo_Qwen3_4B \
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
  actor_rollout_ref.model.use_fused_kernels=False
```

GRPO는 아래 SDPO/SRPO/RLSD/RLRT 전용 self-distillation option을 붙이지 않는다.

```text
actor_rollout_ref.actor.self_distillation.*
actor_rollout_ref.actor.policy_loss.loss_mode=sdpo/srpo/rlsd/rlrt
```

## SRPO

SRPO는 `srpo` config를 쓰거나 `actor_rollout_ref.actor.policy_loss.loss_mode=srpo`를 override한다. 현재 config는 SDPO처럼 distillation context를 만들고, GRPO와 SDPO-style distillation routing을 섞는다.

기본 실행 형태:

```bash
bash training/verl_training.sh \
  qwen3gen-chemistry-SRPO_TR-Qwen-Qwen3-4B-repro-200 \
  srpo \
  datasets/sciknoweval/chemistry \
  max_model_len=10240 \
  data.train_files=[/workspace/SIPO/datasets/sciknoweval/chemistry/train.parquet] \
  data.val_files=[/workspace/SIPO/datasets/sciknoweval/chemistry/test.parquet] \
  data.train_batch_size=32 \
  data.train_max_samples=6400 \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  data.apply_chat_template_kwargs.enable_thinking=false \
  trainer.n_gpus_per_node=8 \
  trainer.total_training_steps=200 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.save_best_checkpoint=True \
  trainer.best_checkpoint_metric=val-aux/sciknoweval/reward/mean@16 \
  trainer.best_checkpoint_mode=max \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=/workspace/SDPO-new-clean/checkpoints/datasets/sciknoweval/chemistry/qwen3gen-chemistry-SRPO_TR-repro-200 \
  custom_reward_function.path=/workspace/SDPO-new-clean/verl/utils/reward_score/feedback/__init__.py \
  +ray_kwargs.ray_init._temp_dir=/tmp/ray_new_q3g_chemistry_srpo_Qwen3_4B \
  +ray_kwargs.ray_init.include_dashboard=False \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
  actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
  actor_rollout_ref.actor.self_distillation.alpha=0.5 \
  actor_rollout_ref.actor.self_distillation.teacher_regularization=trust-region \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.1 \
  actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
  actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting=true \
  actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting_temperature=1.0
```

현재 예약된 SRPO run은 dynamic weighting을 켠다:

```bash
actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting=true \
actor_rollout_ref.actor.self_distillation.srpo_dynamic_weighting_temperature=1.0
```

## RLSD / RLRT

RLSD/RLRT는 SDPO처럼 KL/JSD loss를 직접 더하지 않는다. teacher/student log-ratio로 GRPO advantage를 token-wise reweight한다.

RLSD:

```text
log_ratio = teacher_log_prob - student_log_prob
weight = exp(sign(advantage) * log_ratio)
weight = clip(weight, 1 - eps_w, 1 + eps_w)
advantage = advantage * ((1 - lambda) + lambda * weight)
```

RLRT:

```text
log_ratio = student_log_prob - teacher_log_prob
```

RLRT는 correct rollout에만 reweighting을 적용한다. incorrect rollout은 원래 GRPO advantage를 유지한다.

공통 옵션:

```bash
actor_rollout_ref.actor.self_distillation.token_reweight_lambda=0.5 \
actor_rollout_ref.actor.self_distillation.token_reweight_eps_w=0.2 \
actor_rollout_ref.actor.self_distillation.token_reweight_decay_steps=null
```

RLSD 실행은 SRPO 예시에서 config 이름과 exp name만 바꾼다.

```bash
bash training/verl_training.sh \
  qwen3gen-chemistry-RLSD_TR-Qwen-Qwen3-4B-repro-200 \
  rlsd \
  datasets/sciknoweval/chemistry \
  ...same common arguments...
```

RLRT 실행:

```bash
bash training/verl_training.sh \
  qwen3gen-chemistry-RLRT_TR-Qwen-Qwen3-4B-repro-200 \
  rlrt \
  datasets/sciknoweval/chemistry \
  ...same common arguments...
```

RLSD/RLRT에서도 `teacher_regularization=trust-region`을 그대로 쓸 수 있다. 이때 ref/student 모두 distillation context가 붙은 입력을 보고, teacher logits는 둘을 logit-space로 섞어서 만든다.

## Monitoring

현재 queue PID 확인:

```bash
cat logs/qwen3-4b-chemistry-new-sdpo-then-cmoe05-after.pid
cat logs/qwen3-4b-tooluse-sdpo-variants-after.pid
cat logs/qwen3-4b-all-datasets-plain-sdpo-after.pid
```

현재 프로세스 확인:

```bash
pgrep -af 'queue_new_chemistry|queue_new_tooluse|queue_new_all_datasets|main_ppo'
```

최신 step 확인:

```bash
grep -aE 'Training Progress|step:[0-9]+|Saving new best|val-aux/.*/reward/mean@16' \
  logs/<queue-log>.log | tail -80
```

JSD JSONL 요약:

```bash
/workspace/SIPO/.venv/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("checkpoints/datasets/sciknoweval/chemistry/<run_name>/jsd_histograms.jsonl")
print(p, p.exists(), round(p.stat().st_size / 1024 / 1024, 2) if p.exists() else None)
if p.exists():
    for line in p.read_text().splitlines()[-4:]:
        d = json.loads(line)
        d.pop("values", None)
        print(d)
PY
```

## Expected Runtime

Observed rough runtime on the current H200 8-GPU node:

```text
chemistry 150-step run: about 2 to 2.5 hours, depending on validation/save cost
tooluse 200-step run: about 3 to 3.7 hours
science plain SDPO 200-step run: dataset-dependent, roughly 2.5 to 4 hours each
```

Long queues should be run through `setsid` so they survive shell disconnects:

```bash
setsid bash -c 'RUN_SUFFIX=-myrun bash scripts/queue_new_tooluse_sdpo_after_pid.sh <WAIT_PID>' \
  >/dev/null 2>&1 &
```

## Notes

- `distillation_topk=100` means top-k distillation is active.
- `distillation_add_tail=True` by default, so top-k loss is computed over top-k tokens plus one tail bucket.
- SDPO+mix uses detached student/teacher target distributions. The target side is stop-gradient.
- MoE mixes probabilities; PoE mixes log-probabilities/geometric experts then normalizes.
- TR `teacher_update_rate=0.1` does not mean EMA update 0.1 when `teacher_regularization=trust-region`; it means logit interpolation coefficient 0.1.
- `use_fused_kernels=False` is required for trust-region teacher because logits must be accessible.
