from typing import Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from nn_lib import Tensor
from nn_lib.mdl import Module, Loss
from nn_lib.optim import Optimizer
from nn_lib.data import Dataloader


class ModelTrainer(Module):
    """
    A helper class for manipulating a neural network training and validation
    """

    def __init__(self, model: Module, loss_function: Loss, optimizer: Optimizer):
        """
        Create a neural network
        :param model: a model to train
        :param loss_function: loss function module to use for training
        :param optimizer: optimizer to use for training
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, x: Tensor) -> Tensor:
        predictions = self.model(x)

        return predictions

    def train(self, train_dataloader: Dataloader, n_epochs: int) -> None:

        progress_bar = tqdm(range(n_epochs * len(train_dataloader)))
        for i_epoch in range(n_epochs):
            for data_batch, label_batch in train_dataloader:
                # print(label_batch.shape)

                _, loss_value = self._train_step(data_batch, label_batch)

                progress_bar.update(1)

    #               progress_bar.desc = f'Training. Epoch: {i_epoch + 1}. Loss: {loss_value.data:.4f}'

    def _train_step(self, data_batch: Tensor, label_batch: Tensor) -> Tuple[Tensor, Tensor]:
        optimizer = self.optimizer
        loss = self.loss_function
        model = self.model

        optimizer.zero_grad()

        preds = model(data_batch)

        loss = loss(preds, label_batch)
        print(loss)

        loss.backward()
        optimizer.step()
        return preds, label_batch

    def validate(self, test_dataloader: Dataloader) -> Tuple[np.ndarray, float, float]:

        n_correct_predictions = 0
        n_predictions = 0
        loss_values_sum = 0
        predictions = []
        for data_batch, label_batch in tqdm(test_dataloader, desc='Validating'):
            prediction_logit_batch = self.model(data_batch)

            positive_predictions = prediction_logit_batch.data > 0
            # positive_predictions = list(map(lambda x: 1 if x == [True] else 0, positive_predictions))

            correct_predictions = positive_predictions == label_batch.data

            n_correct_predictions += correct_predictions.sum()

            n_predictions += len(data_batch.data)

            loss_value = self.loss_function(prediction_logit_batch, label_batch)

            loss_values_sum += loss_value.data

            predictions.extend(positive_predictions)

        predictions = np.array(predictions, np.bool)
        accuracy = n_correct_predictions / n_predictions
        mean_loss = loss_values_sum / n_predictions

        return predictions, accuracy, mean_loss
