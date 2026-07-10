# Current Queue Hyperparameters

작성 시점: 2026-07-10

이 문서는 현재 `/workspace/SDPO-new-clean`에서 실행 중이거나 예약된 run들의 하이퍼파라미터를 묶음별로 정리한 것이다.

## Environment

```text
repo: /workspace/SDPO-new-clean
venv: /workspace/SIPO/.venv
python: /workspace/SIPO/.venv/bin/python
PYTHONPATH: /workspace/SDPO-new-clean
WORKSPACE_DIR: /workspace/SIPO
SKIP_INSTALL: true
CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
NUM_GPUS: 8
WANDB_ENTITY: seongryongjung-chung-ang-university
```

공통 실행 전 Ray 정리:

```bash
/workspace/SIPO/.venv/bin/python -m ray.scripts.scripts stop --force
```

## Common Model / Rollout / Optimizer Settings

아래 값은 별도 표기가 없으면 모든 SDPO/SRPO 예약 run에 공통이다.

```text
model: Qwen/Qwen3-4B
n_gpus_per_node: 8
max_model_len: 10240
max_prompt_length: 2048
max_response_length: 8192
train_batch_size: 32
ppo_mini_batch_size: 32
optim.lr_warmup_steps: 10
optim.weight_decay: 0.01
grad_clip: 1.0
clip_ratio_low: 0.2
clip_ratio_high: 0.28
rollout.n: 8
rollout.temperature: 1.0
rollout.top_p: 1.0
rollout.max_model_len: 10240
rollout.max_num_batched_tokens: 10240
rollout.gpu_memory_utilization: 0.8
rollout.calculate_log_probs: true
val_kwargs.n: 16
val_kwargs.temperature: 0.6
val_kwargs.top_p: 0.95
val_kwargs.do_sample: true
norm_adv_by_std_in_grpo: false
rollout_correction.rollout_is: token
rollout_correction.rollout_is_threshold: 2.0
use_fused_kernels: false
enable_thinking: false
```

## Common Evaluation / Saving

```text
trainer.val_before_train: false
trainer.test_freq: 5
trainer.save_freq: -1
trainer.save_best_checkpoint: true
trainer.best_checkpoint_mode: max
trainer.max_actor_ckpt_to_keep: 1
trainer.max_critic_ckpt_to_keep: 1
```

Best metric:

```text
sciknoweval datasets: val-aux/sciknoweval/reward/mean@16
tooluse:              val-aux/tooluse/reward/mean@16
```

즉 `step 5, 10, 15, ...` 마다 평가하고, best metric이 좋아질 때만 checkpoint를 저장한다.

## Common SDPO / SRPO Distillation Settings

```text
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

TR teacher:

```text
teacher_logits = 0.9 * ref_logits + 0.1 * current_student_logits
```

## Dataset Paths

현재 실행 중인 chemistry process는 시작 당시 인자로 `/workspace/L2T-sdpo-mix-pr/datasets/sciknoweval/chemistry/*.parquet`를 받았다.

커밋된 재현용 스크립트와 앞으로 새로 실행할 watcher는 `/workspace/SIPO/datasets`를 사용한다.

```text
chemistry train: /workspace/SIPO/datasets/sciknoweval/chemistry/train.parquet
chemistry val:   /workspace/SIPO/datasets/sciknoweval/chemistry/test.parquet
physics train:   /workspace/SIPO/datasets/sciknoweval/physics/train.parquet
physics val:     /workspace/SIPO/datasets/sciknoweval/physics/test.parquet
biology train:   /workspace/SIPO/datasets/sciknoweval/biology/train.parquet
biology val:     /workspace/SIPO/datasets/sciknoweval/biology/test.parquet
material train:  /workspace/SIPO/datasets/sciknoweval/material/train.parquet
material val:    /workspace/SIPO/datasets/sciknoweval/material/test.parquet
tooluse train:   /workspace/SIPO/datasets/tooluse/train.parquet
tooluse val:     /workspace/SIPO/datasets/tooluse/test.parquet
```

## Queue Order

현재 예약 순서:

```text
1. chemistry SDPO+mix variants, 150 steps
2. tooluse SDPO/plain+mix variants, 200 steps
3. chemistry/physics/biology/material plain SDPO, 200 steps
4. chemistry/physics/biology/material/tooluse SRPO, 200 steps
```

PID chain:

```text
chemistry queue -> tooluse variants watcher -> science plain SDPO watcher -> all-dataset SRPO watcher
```

## 1. Chemistry SDPO+Mix Variants

Script:

```text
scripts/queue_new_chemistry_sdpo_then_cmoe05_after_pid.sh
```

Queue PID at the time of writing:

```text
4097033
```

Training PID for current active run:

```text
4097116
```

Shared settings:

```text
dataset: chemistry
config: sdpo
policy_loss.loss_mode: sdpo
total_training_steps: 150
train_batch_size: 32
train_max_samples: 4800
optim.lr: 1e-5
distillation_topk: 100
alpha: 0.5
teacher_regularization: trust-region
teacher_update_rate: 0.1
is_clip: 2.0
test_freq: 5
save_best_checkpoint: true
best metric: val-aux/sciknoweval/reward/mean@16
```

Variants:

| Order | Run suffix | correct target | incorrect target |
| --- | --- | --- | --- |
| 1 | `sdpo-cmoe03-150clean-jsdhist3` | MoE, student weight `0.3` | MoE, student weight `0.0` |
| 2 | `sdpo-cpoe03-150clean-jsdhist3` | PoE, student weight `0.3` | PoE, student weight `0.0` |
| 3 | `sdpo-imoe03-150clean-jsdhist3` | MoE, student weight `0.0` | MoE, student weight `0.3` |
| 4 | `sdpo-ipoe03-150clean-jsdhist3` | PoE, student weight `0.0` | PoE, student weight `0.3` |

Notes:

```text
MoE target = beta * stopgrad(student_dist) + (1 - beta) * stopgrad(teacher_dist)
PoE target = normalize(beta * student_logp + (1 - beta) * teacher_logp)
```

## 2. Tooluse SDPO Plain + Mix Variants

Script:

```text
scripts/queue_new_tooluse_sdpo_after_pid.sh
```

Current watcher PID:

```text
4136768
```

Waits for:

```text
4097033
```

Shared settings:

```text
dataset: tooluse
config: sdpo
policy_loss.loss_mode: sdpo
total_training_steps: 200
train_batch_size: 32
train_max_samples: 6400
optim.lr: 1e-5
distillation_topk: 100
alpha: 0.5
teacher_regularization: trust-region
teacher_update_rate: 0.1
is_clip: 2.0
test_freq: 5
save_best_checkpoint: true
best metric: val-aux/tooluse/reward/mean@16
```

Variants:

| Order | Run suffix | correct target | incorrect target |
| --- | --- | --- | --- |
| 1 | `sdpo-200clean-jsdhist3` | no SDPO target mix | no SDPO target mix |
| 2 | `sdpo-cmoe03-200clean-jsdhist3` | MoE, student weight `0.3` | MoE, student weight `0.0` |
| 3 | `sdpo-cpoe03-200clean-jsdhist3` | PoE, student weight `0.3` | PoE, student weight `0.0` |
| 4 | `sdpo-imoe03-200clean-jsdhist3` | MoE, student weight `0.0` | MoE, student weight `0.3` |
| 5 | `sdpo-ipoe03-200clean-jsdhist3` | PoE, student weight `0.0` | PoE, student weight `0.3` |

## 3. Science Plain SDPO

Script:

```text
scripts/queue_new_all_datasets_plain_sdpo_after_pid.sh
```

Current watcher PID:

```text
4136779
```

Waits for:

```text
4136768
```

Datasets:

```text
chemistry
physics
biology
material
```

Tooluse is intentionally excluded from this queue because tooluse plain SDPO already runs in the previous tooluse queue.

Shared settings:

```text
config: sdpo
policy_loss.loss_mode: sdpo
total_training_steps: 200
train_batch_size: 32
train_max_samples: 6400
optim.lr: 1e-5
distillation_topk: 100
alpha: 0.5
teacher_regularization: trust-region
teacher_update_rate: 0.1
is_clip: 2.0
test_freq: 5
save_best_checkpoint: true
best metric: val-aux/sciknoweval/reward/mean@16
```

Order:

| Order | Dataset | Run suffix |
| --- | --- | --- |
| 1 | chemistry | `sdpo-200clean-aftertooluse` |
| 2 | physics | `sdpo-200clean-aftertooluse` |
| 3 | biology | `sdpo-200clean-aftertooluse` |
| 4 | material | `sdpo-200clean-aftertooluse` |

## 4. All-Dataset SRPO

Script:

```text
scripts/queue_new_all_datasets_srpo_after_pid.sh
```

Current watcher PID:

```text
4136787
```

Waits for:

```text
4136779
```

Datasets:

```text
chemistry
physics
biology
material
tooluse
```

Shared settings:

```text
config: srpo
policy_loss.loss_mode: srpo
total_training_steps: 200
train_batch_size: 32
train_max_samples: 6400
optim.lr: 5e-6
optim.lr_warmup_steps: 10
optim.weight_decay: 0.01
grad_clip: 1.0
rollout.n: 8
distillation_topk: 100
alpha: 0.5
teacher_regularization: trust-region
teacher_update_rate: 0.1
is_clip: 2.0
srpo_dynamic_weighting: true
srpo_dynamic_weighting_temperature: 1.0
test_freq: 5
save_best_checkpoint: true
```

Best metrics:

```text
chemistry/physics/biology/material: val-aux/sciknoweval/reward/mean@16
tooluse:                         val-aux/tooluse/reward/mean@16
```

Order:

| Order | Dataset | Run suffix |
| --- | --- | --- |
| 1 | chemistry | `srpo-200clean-aftersdpo200` |
| 2 | physics | `srpo-200clean-aftersdpo200` |
| 3 | biology | `srpo-200clean-aftersdpo200` |
| 4 | material | `srpo-200clean-aftersdpo200` |
| 5 | tooluse | `srpo-200clean-aftersdpo200` |

Actual run names include:

```text
SRPO_TR
tr0.1
dwtrue
train32
rollout8
lr5e-6
vllm0.8
newrepo
srpo-200clean
```

## Logs

Queue logs:

```text
chemistry queue:
logs/qwen3-4b-chemistry-new-sdpo-then-cmoe05-after-now-remaining-20260710-143415.log

tooluse queue:
logs/qwen3-4b-tooluse-sdpo-variants-after-4097033-20260710-164502.log

science plain SDPO queue:
logs/qwen3-4b-all-datasets-plain-sdpo-after-4136768-20260710-164502.log

SRPO queue:
logs/qwen3-4b-all-datasets-srpo-after-4136779-20260710-164502.log
```

PID files:

```text
logs/qwen3-4b-chemistry-new-sdpo-then-cmoe05-after.pid
logs/qwen3-4b-tooluse-sdpo-variants-after.pid
logs/qwen3-4b-all-datasets-plain-sdpo-after.pid
logs/qwen3-4b-all-datasets-srpo-after.pid
```

JSD histogram JSONL:

```text
<checkpoint_dir>/jsd_histograms.jsonl
```

Example:

```text
checkpoints/datasets/sciknoweval/chemistry/<run_name>/jsd_histograms.jsonl
checkpoints/datasets/tooluse/<run_name>/jsd_histograms.jsonl
```

## Monitoring Commands

Check active queues:

```bash
pgrep -af 'queue_new_chemistry|queue_new_tooluse|queue_new_all_datasets|main_ppo'
```

Check current queue log path:

```bash
cat logs/qwen3-4b-chemistry-new-sdpo-then-cmoe05-after.logpath
cat logs/qwen3-4b-tooluse-sdpo-variants-after.logpath
cat logs/qwen3-4b-all-datasets-plain-sdpo-after.logpath
cat logs/qwen3-4b-all-datasets-srpo-after.logpath
```

Check recent training progress:

```bash
grep -aE 'Training Progress|step:[0-9]+|Saving new best|val-aux/.*/reward/mean@16' <queue-log> | tail -80
```

Check latest best checkpoint:

```bash
cat <checkpoint_dir>/latest_checkpointed_iteration.txt
```

