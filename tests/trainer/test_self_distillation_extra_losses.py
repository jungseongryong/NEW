import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_self_distillation_loss,
    compute_self_distillation_reweighted_advantages,
    compute_srpo_policy_loss,
    is_self_distillation_loss_mode,
    is_self_distillation_reweight_loss_mode,
)
from verl.workers.config import SelfDistillationConfig


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


def test_self_distillation_loss_modes():
    assert is_self_distillation_loss_mode("sdpo")
    assert is_self_distillation_loss_mode("srpo")
    assert is_self_distillation_loss_mode("rlsd")
    assert is_self_distillation_loss_mode("rlrt")
    assert is_self_distillation_reweight_loss_mode("rlsd")
    assert is_self_distillation_reweight_loss_mode("rlrt")


def test_rlsd_reweights_with_teacher_over_student_ratio():
    student_log_probs = torch.tensor([[-2.0, -2.0]])
    teacher_log_probs = torch.tensor([[-1.0, -3.0]])
    advantages = torch.tensor([[2.0, -2.0]])
    response_mask = torch.ones_like(advantages)
    cfg = {"token_reweight_lambda": 0.5, "token_reweight_eps_w": 0.2}

    refined, metrics = compute_self_distillation_reweighted_advantages(
        loss_mode="rlsd",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=torch.tensor([1.0]),
    )

    expected_modulator = 0.5 + 0.5 * 1.2
    assert torch.allclose(refined, advantages * expected_modulator)
    assert metrics["self_distillation/token_reweight_w_clip_frac"] == 1.0


def test_rlrt_reverses_teacher_signal_and_gates_to_correct_rollouts():
    student_log_probs = torch.tensor([[-1.0, -3.0], [-1.0, -3.0]])
    teacher_log_probs = torch.tensor([[-2.0, -2.0], [-2.0, -2.0]])
    advantages = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    response_mask = torch.ones_like(advantages)
    cfg = {"token_reweight_lambda": 1.0, "token_reweight_eps_w": 10.0}

    refined, metrics = compute_self_distillation_reweighted_advantages(
        loss_mode="rlrt",
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=torch.tensor([1.0, 1.0]),
        self_distillation_correct_mask=torch.tensor([1.0, 0.0]),
    )

    expected = torch.tensor(
        [
            [2.0 * torch.exp(torch.tensor(1.0)).item(), 2.0 * torch.exp(torch.tensor(-1.0)).item()],
            [2.0, 2.0],
        ]
    )
    assert torch.allclose(refined, expected)
    assert metrics["self_distillation/token_reweight_is_rlrt"] == 1.0


def test_srpo_routes_correct_samples_to_grpo_and_context_failures_to_sdpo():
    old_log_prob = torch.tensor([[0.0], [0.0], [3.0]])
    log_prob = torch.tensor([[0.0], [0.0], [3.0]])
    teacher_log_probs = torch.tensor([[0.0], [0.0], [2.0]])
    advantages = torch.tensor([[2.0], [4.0], [0.0]])
    response_mask = torch.ones_like(advantages)
    actor_cfg = AttrDict(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None, clip_ratio_c=3.0)
    sd_cfg = AttrDict(full_logit_distillation=False, alpha=1.0, is_clip=None, srpo_dynamic_weighting=False)

    loss, metrics = compute_srpo_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        teacher_log_probs=teacher_log_probs,
        self_distillation_config=sd_cfg,
        self_distillation_mask=torch.tensor([0.0, 0.0, 1.0]),
        self_distillation_correct_mask=torch.tensor([1.0, 1.0, 0.0]),
        config=actor_cfg,
    )

    assert torch.allclose(loss, torch.tensor(-1.0))
    assert metrics["srpo/grpo_loss"] == -3.0
    assert metrics["srpo/sdpo_loss"] == 3.0


def test_sdpo_can_mix_teacher_target_with_moe_and_poe():
    student_all_log_probs = torch.tensor([[[0.8, 0.2]]]).log()
    teacher_all_log_probs = torch.tensor([[[0.2, 0.8]]]).log()

    for mode in ("moe", "poe"):
        sd_cfg = AttrDict(
            full_logit_distillation=True,
            distillation_topk=None,
            alpha=0.0,
            sdpo_teacher_mix_student_weight=0.25,
            sdpo_teacher_mix_mode=mode,
            is_clip=None,
            srpo_dynamic_weighting=False,
        )
        loss, metrics = compute_self_distillation_loss(
            student_log_probs=student_all_log_probs[:, :, 0],
            teacher_log_probs=teacher_all_log_probs[:, :, 0],
            response_mask=torch.ones(1, 1),
            self_distillation_config=sd_cfg,
            student_all_log_probs=student_all_log_probs,
            teacher_all_log_probs=teacher_all_log_probs,
        )

        if mode == "moe":
            mixed_target = (0.25 * student_all_log_probs.exp() + 0.75 * teacher_all_log_probs.exp()).log()
        else:
            mixed_target = 0.25 * student_all_log_probs + 0.75 * teacher_all_log_probs
            mixed_target = mixed_target - torch.logsumexp(mixed_target, dim=-1, keepdim=True)
        expected = torch.nn.functional.kl_div(
            student_all_log_probs,
            mixed_target,
            reduction="sum",
            log_target=True,
        )
        assert torch.allclose(loss, expected)
        assert metrics[f"self_distillation/sdpo_teacher_mix_mode_{mode}"] == 1.0


def test_self_distillation_config_validates_sdpo_teacher_mix():
    with pytest.raises(ValueError, match="sdpo_teacher_mix_student_weight"):
        SelfDistillationConfig(sdpo_teacher_mix_student_weight=1.1)
    with pytest.raises(ValueError, match="sdpo_teacher_mix_mode"):
        SelfDistillationConfig(sdpo_teacher_mix_mode="bad")
