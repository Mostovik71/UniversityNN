from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Log(Function):
    """
    Natural logarithm function (e based)
    """
    def __init__(self,arg):
        self.arg=arg

    def forward(self) -> np.ndarray:
        """
        Compute logarithm of the argument, i.e. log(self.args[0].data)

        :return: logarithm of the argument
        """
        return np.log(self.arg)
    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the logarithm argument

        :param grad_output: gradient over the result of the logarithm function
        :return: a tuple with a single value representing the gradient over the logarithm argument
        """
        raise NotImplementedError   # TODO: implement me as an exercise
if __name__ == '__main__':
    print(Log(2).forward())