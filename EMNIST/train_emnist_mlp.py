from nn_lib.mdl import BCELoss, CELoss
from nn_lib.optim import SGD
from nn_lib.data import Dataloader
from EMNIST.mlp_classifier import MLPClassifier

from EMNIST.model_trainer import ModelTrainer

from EMNIST.emnist_dataset import EMnistDataset


def main(n_samples, n_epochs, hidden_layer_sizes):
    mlp_model = MLPClassifier(in_features=784, hidden_layer_sizes=hidden_layer_sizes)
    print(f'Created the following binary MLP classifier:\n{mlp_model}')

    loss_fn = CELoss()

    optimizer = SGD(mlp_model.parameters(), lr=1e-4, weight_decay=5e-4)

    model_trainer = ModelTrainer(mlp_model, loss_fn, optimizer)

    train_dataset = EMnistDataset(n_samples=n_samples, seed=12)

    val_dataset = EMnistDataset(n_samples=n_samples, seed=15)

    train_dataloader = Dataloader(train_dataset, batch_size=100, shuffle=True, drop_last=True)

    val_dataloader = Dataloader(val_dataset, batch_size=100, shuffle=False, drop_last=False)

    model_trainer.train(train_dataloader, n_epochs=n_epochs)

    train_predictions, train_accuracy, train_mean_loss = model_trainer.validate(
        train_dataloader)
    print(f'Train accuracy: {train_accuracy:.4f}')
    print(f'Train loss: {train_mean_loss:.4f}')

    val_predictions, val_accuracy, val_mean_loss = model_trainer.validate(val_dataloader)
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print(f'Validation loss: {val_mean_loss:.4f}')


if __name__ == '__main__':
    main(n_samples=100, n_epochs=100, hidden_layer_sizes=(20,))
    # main(n_samples=1000, n_epochs=150, hidden_layer_sizes=(30, 30))
    # main(n_samples=1000, n_epochs=100, hidden_layer_sizes=(100,))
