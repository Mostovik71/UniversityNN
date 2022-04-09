from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Inv(Function):
    """
    Multiplication inverse function
    """
    def _inv(self, data: np.ndarray):
        return 1 / data

    def forward(self) -> np.ndarray:
        """
        Compute multiplicative inverse of the argument, i.e. (self.args[0].data) ^ -1

        :return: inverse of the argument
        """
        return self._inv(self.args[0].data)

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the inversion argument

        :param grad_output: gradient over the result of the invert function
        :return: a tuple with a single value representing the gradient over the inversion argument
        """

        inv = self._inv(self.args[0].data)
        return (grad_output * (-inv * inv),)




