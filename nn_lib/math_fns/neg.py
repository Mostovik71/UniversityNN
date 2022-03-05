from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Neg(Function):
    """
    Negation function (additive inverse)
    """
    def __init__(self, arg1):
        self.arg1 = arg1
    def forward(self) -> np.ndarray:
        """
        Take negative of the argument, i.e. -self.args[0].data

        :return: negative of the argument
        """
        return -self.arg1

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the negation argument

        :param grad_output: gradient over the result of the negation
        :return: a tuple with a single value representing the gradient over the negation argument
        """
        raise NotImplementedError   # TODO: implement me as an exercise
if __name__ == '__main__':
    print(Neg(1).forward())