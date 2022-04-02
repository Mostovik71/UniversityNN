import numpy as np

from nn_lib.mdl.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class CELoss(Loss):


    # In order to avoid over- or underflow we clip prediction logits into [-MAX_LOG, MAX_LOG]
    MAX_LOG = 50

    def _clip(self, value: Tensor) -> Tensor:
        return F.clip(value, Tensor(-self.MAX_LOG, requires_grad=True), Tensor(self.MAX_LOG, requires_grad=True))
    def forward(self, prediction_logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute a loss Tensor based on logit predictions and ground truth labels

        :param prediction_logits: prediction logits returned by a model (i.e. sigmoid argument) of shape (B,)
        :param target: binary ground truth labels of shape (B,)
        :return: a loss Tensor; if reduction is True, returns a scalar, otherwise a Tensor of shape (B,) -- loss value
            per batch element
        """
        a = self._clip(prediction_logits)

        losses = (Tensor(1) - target) * a + F.log(Tensor(1) + F.exp(-a))
        return F.reduce(losses) if self.reduce else losses


