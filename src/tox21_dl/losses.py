"""Masked binary cross-entropy and endpoint class weights for Tox21."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F


def compute_pos_weight(y_train, cap=10.0):                                       # positive(1) 클래스에 더 큰 중요도를 주기 위한 가중치 (pos_weight = n_negative / n_positive)
    """Return one float32 positive-class weight per endpoint.                     # neg가 pos보다 9배 많으면, pos를 9배 중요하게 보겠다.
                                                                                     # 왜 필요하냐? 모델이 그냥 학습하면 "0이 훨씬 많네? 그냥 다 0으로 예측하면 되겠네
    Only training labels (0, 1 or NaN) are used. Missing labels are ignored.          # pos_weight이 있으면 "positive 틀림 → 더 크게 벌점 / negative 틀림 → 덜 벌점
    cap=None: use the negative/positive ratio without an upper limit.                  # => binary classification + imbalance 있을 때 거의 필수
    cap=1: explicitly disable class weighting (all weights equal one).                # cap은 너무 과한 가중치를 막기위한 한계점, 리밋을 걸어두는것
    cap>1: limit the ratio from above; ratios below one remain unchanged.              # e.g. compute_pos_weight(y_tr, cap=10)	최대 weight 10 vs. cap=1 -> Class weighting 미적용
    Single-class or unlabeled endpoints receive a neutral weight of one.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.ndim != 2 or 0 in y_train.shape:
        raise ValueError("y_train must be a nonempty 2-D label array.")
    if not np.all(np.isnan(y_train) | (y_train == 0) | (y_train == 1)):
        raise ValueError("Training labels must be 0, 1 or NaN.")
    if cap is not None:
        cap = float(cap)
        if not np.isfinite(cap) or cap < 1:
            raise ValueError("cap must be None or a finite number >= 1.")

    n_pos = (y_train == 1).sum(axis=0)          # positive / negative 개수 계산
    n_neg = (y_train == 0).sum(axis=0)
    both_classes = (n_pos > 0) & (n_neg > 0)
    weights = np.ones(y_train.shape[1], dtype=np.float64)
    if not both_classes.all():
        warnings.warn(
            f"Endpoints without both classes (column indices "
            f"{np.flatnonzero(~both_classes).tolist()}) receive pos_weight=1.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cap != 1:
        weights[both_classes] = n_neg[both_classes] / n_pos[both_classes]
        if cap is not None:
            weights = np.minimum(weights, cap)      # weight 계산
    return torch.tensor(weights, dtype=torch.float32)


def masked_bce_with_logits_loss(logits, targets, mask, pos_weight=None):        # NaN(라벨 없는 부분)을 무시하고 BCE loss를 계산
    """Mean weighted BCE over valid labels only; return a scalar tensor.

    logits, targets, mask: (batch_size, n_tasks), with a binary mask.
    targets may contain NaN at masked positions or use zero placeholders.
    pos_weight: optional tensor with shape (n_tasks,).
    For an all-missing batch, return a differentiable zero. The training
    loop should skip the forward/optimizer step for such a batch.
    """
    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets and mask must have the same 2-D shape.")
    if logits.device != targets.device or logits.device != mask.device:
        raise ValueError("logits, targets and mask must be on the same device.")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0 or 1.")

    valid = mask.bool()
    valid_logits = logits[valid]
    valid_targets = targets[valid].to(dtype=logits.dtype)
    if not torch.all((valid_targets == 0) | (valid_targets == 1)):
        raise ValueError("Unmasked targets must be 0 or 1.")

    valid_weights = None
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        if pos_weight.shape != (logits.shape[1],):
            raise ValueError("pos_weight must have one value per endpoint.")
        if not torch.all(torch.isfinite(pos_weight) & (pos_weight > 0)):
            raise ValueError("pos_weight must contain finite positive values.")
        # Match each selected label with its own endpoint weight.
        valid_weights = pos_weight.unsqueeze(0).expand_as(logits)[valid]

    if valid_logits.numel() == 0:
        return valid_logits.sum()

    # Select valid entries before BCE so masked NaN targets never enter it.
    return F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        pos_weight=valid_weights,
        reduction="mean",
    )
