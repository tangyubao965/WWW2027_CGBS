from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_cross_entropy(outputs) -> torch.Tensor:
    """
    Return the standard seq2seq loss from Hugging Face model outputs.
    """
    if outputs.loss is None:
        raise ValueError("Model outputs do not contain loss. Please pass labels to the model.")
    return outputs.loss


def token_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Token-level KL distillation loss.

    Args:
        student_logits:
            Student logits with shape [batch_size, seq_len, vocab_size].
        teacher_logits:
            Teacher logits with shape [batch_size, seq_len, vocab_size].
        labels:
            Target token ids with shape [batch_size, seq_len], where -100 is ignored.
        temperature:
            Distillation temperature.

    Returns:
        Scalar KL loss over non-ignored target positions.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student_logits and teacher_logits must have the same shape, "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )

    if labels.shape != student_logits.shape[:2]:
        raise ValueError(
            f"labels shape must match first two dims of logits, got "
            f"{tuple(labels.shape)} and {tuple(student_logits.shape)}."
        )

    mask = labels.ne(-100).to(student_logits.dtype)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)

    loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    return loss * (temperature ** 2)


def hidden_mse_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Hidden-state distillation loss.

    Args:
        student_hidden:
            Student hidden states with shape [batch_size, seq_len, hidden_size].
        teacher_hidden:
            Teacher hidden states with shape [batch_size, seq_len, hidden_size].
        attention_mask:
            Optional mask with shape [batch_size, seq_len].

    Returns:
        Scalar MSE loss.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"student_hidden and teacher_hidden must have the same shape, "
            f"got {tuple(student_hidden.shape)} and {tuple(teacher_hidden.shape)}."
        )

    mse = (student_hidden - teacher_hidden) ** 2
    mse = mse.mean(dim=-1)

    if attention_mask is not None:
        mask = attention_mask.to(mse.dtype)
        return (mse * mask).sum() / mask.sum().clamp_min(1.0)

    return mse.mean()


def combine_losses(loss_dict: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """
    Combine multiple losses with weights.

    Args:
        loss_dict:
            Mapping from loss name to scalar tensor.
        weights:
            Mapping from loss name to scalar weight.

    Returns:
        Weighted loss.
    """
    total = None

    for name, loss in loss_dict.items():
        weight = float(weights.get(name, 1.0))
        weighted = weight * loss
        total = weighted if total is None else total + weighted

    if total is None:
        raise ValueError("loss_dict is empty.")

    return total