import numpy as np

from nn_lib.mdl.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class BCELoss(Loss):
    """
    Binary cross entropy loss
    Similar to this https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """

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



        # if self.reduce==True:
        #     if target.shape==():
        #         prediction_logits=Tensor(1)/(Tensor(1)+F.exp(-prediction_logits))
        #         loss=-(target*F.log(prediction_logits)+(Tensor(1)-target)*F.log(Tensor(1)-prediction_logits))
        #     else:
        #         prediction_logits = [Tensor(1) / (Tensor(1) + F.exp(-i)) for i in prediction_logits]
        #
        #         loss = np.sum([-(i * F.log(j) + (Tensor(1) - i) * F.log(Tensor(1) - j)) for i, j in
        #                 zip(target, prediction_logits)])/Tensor(len(target.data))
        #         print(loss)
        #
        #     return loss
        # elif self.reduce==False:
        #
        #     prediction_logits = [Tensor(1) / (Tensor(1) + F.exp(-i)) for i in prediction_logits]
        #     logs1=[]
        #     logs2=[]
        #     for j in (prediction_logits):
        #         logs1.append(F.log(j))
        #         logs2.append(F.log(Tensor(1)-j))
        #     logs1=[self._clip(i) for i in logs1]
        #     logs2 = [self._clip(i) for i in logs2]
        #     losses=[]
        #     for i in range(len(target.data)):
        #         losses.append(-((target[i]*logs1[i])+(Tensor(1)-target[i]))*logs2[i])
        #     for i in losses:
        #         print(i)
            #loss=([-(i*F.log(j)+(Tensor(1)-i)*F.log(Tensor(1)-j)) for i,j in zip(target, prediction_logits)])

            # loss=np.array([i.data for i in loss])
            #
            #
            # return loss
