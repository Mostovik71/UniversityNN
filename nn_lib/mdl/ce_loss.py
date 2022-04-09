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
        prediction_logits = self._clip(prediction_logits)

        x = prediction_logits  # F.softmax(prediction_logits)

        log = F.log(x)
        log = self._clip(log)
        losses = - target * log

        return F.reduce(losses) if self.reduce else losses



if __name__ == '__main__':
    loss = CELoss()
    x = Tensor([0.08, 0.04, 0.8, 0.08])
    y = Tensor([0, 0, 1, 0])
    print(loss(x, y))

