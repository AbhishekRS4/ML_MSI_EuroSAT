import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Union
from torch.nn.modules.loss import _Loss


class FocalLoss(_Loss):
    def __init__(
        self,
        weight: Union[Tensor, None] = None,
        gamma: int = 2.0,
        reduction: str = "mean",
    ):
        """
        ----------
        Attributes
        ----------
        weight: Union[Tensor, None]
            a torch tensor of class weights to be applied during training (default: None)
        gamma: int
            indicates the importance to be given to the misclassified classes (default: 2)
        reduction: str,
            reduction to be applied to the loss function (default: mean)
        """
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, pred_logits: Tensor, target_labels: Tensor, dim: int = 1
    ) -> Tensor:
        """
        ---------
        Arguments
        ---------
        pred_logits: Tensor
            a tensor of predicted logits
        target_labels: Tensor
            a tensor of target labels
        dim: int
            dimension along which the softmax needs to be applied (default: 1)
        """
        log_prob = F.log_softmax(pred_logits, dim=dim)
        prob = torch.exp(log_prob)

        focal_loss = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_labels,
            weight=self.weight,
            reduction=self.reduction,
        )

        return focal_loss
