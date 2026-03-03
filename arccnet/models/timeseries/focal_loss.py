"""Focal Loss implementation for handling imbalanced classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification.

    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard negative examples and down-weight easy examples.

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    (https://arxiv.org/abs/1708.02002)

    Parameters
    ----------
    alpha : float or list/tensor, optional
        Weighting factor in [0, 1] to balance positive/negative examples.
        If list/tensor, should have length num_classes for per-class weighting.
        Default: 0.25
    gamma : float, optional
        Exponent of the modulating factor (1 - p_t)^gamma.
        Higher values focus more on hard examples. Default: 2.0
    reduction : str, optional
        Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'
    weight : tensor, optional
        Manual rescaling weight given to each class.
        If given, has to be a Tensor of size `C`.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", weight=None):
        super().__init__()
        self.alpha = None
        self.gamma = float(gamma)
        self.reduction = reduction

        # Keep class weights as a buffer so Lightning/device transfers move it with the module.
        if weight is None:
            self.register_buffer("weight", None)
        elif isinstance(weight, torch.Tensor):
            self.register_buffer("weight", weight.float())
        else:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))

        # Support scalar alpha or per-class alpha vector.
        if isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha_tensor", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha_tensor", alpha.float())
        else:
            self.alpha = float(alpha)
            self.register_buffer("alpha_tensor", None)

    def forward(self, inputs, targets):
        """
        Calculate focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Model logits, shape (N, C) where N is batch size and C is number of classes
        targets : torch.Tensor
            Ground truth class indices, shape (N,)

        Returns
        -------
        loss : torch.Tensor
            Focal loss
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Get class probabilities
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)

        # Get probability of the true class for each sample
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)

        # Apply focal loss (down-weight easy examples)
        loss = ce * ((1 - p_t) ** self.gamma)

        # Apply alpha weighting if specified
        if self.alpha_tensor is not None:
            alpha_t = self.alpha_tensor
            if alpha_t.device != inputs.device:
                alpha_t = alpha_t.to(inputs.device)
            alpha_t = alpha_t.gather(0, targets)
            loss = alpha_t * loss
        elif self.alpha is not None and self.alpha > 0:
            loss = self.alpha * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
