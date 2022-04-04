from typing import Union, Tuple, List
import numpy as np

from nn_lib import Tensor
from nn_lib.mdl import Module, Linear


class MLPClassifier(Module):

    def __init__(self, in_features: int, hidden_layer_sizes: Union[Tuple[int, ...], List[int]]):

        self.in_features = in_features
        self.hidden_layer_sizes = hidden_layer_sizes

        self._parameters = []
        self.layers = []  # type: List[Linear]
        self._build_layers()

    def parameters(self) -> List[Tensor]:
        result = self._parameters.copy()

        return result

    def _build_layers(self) -> None:

        number_of_layers = len(self.hidden_layer_sizes)
        if number_of_layers == 0:
            return None
        for i in range(0, number_of_layers):
            in_dim = out_dim if i != 0 else self.in_features
            out_dim = self.hidden_layer_sizes[i]

            self._add_layer(in_dim, out_dim, 'relu')

        self._add_layer(out_dim, 10, 'softmax')

    def _add_layer(self, in_dim: int, out_dim: int, activation_fn: str) -> None:

        layer = Linear(in_dim, out_dim, activation_fn)
        self.layers.append(layer)
        self._parameters.append(layer.weight)
        self._parameters.append(layer.bias)

    def forward(self, x: Tensor) -> Tensor:

        for layer in self.layers:
            #print(x)
            x = layer.forward(x)

        return x  # [:, 0]

    def parameter_count(self) -> int:

        result = 0
        for param in self.parameters():
            result += np.prod(param.shape)
        return result

    def __str__(self) -> str:
        result = '\n'.join(map(str, self.layers)) + f'\nTotal number of parameters: {self.parameter_count()}'
        return result
