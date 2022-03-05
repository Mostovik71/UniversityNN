from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Add(Function):
    """
    Addition of two elements (Сложение)
    """

    def __init__(self, *args):
        self.args = args

    def forward(self) -> np.ndarray:
        """
        Add two arguments and return their sum

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: sum of the two arguments
        """

        return sum(self.args)

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over two addition arguments

        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the addition operation
        :return: a tuple of gradients over two addition arguments
        """
        raise NotImplementedError   # TODO: implement me as an exercise

if __name__ == '__main__':# Oh God, it works not only for two elements
    res=Add(1,2,3,4,5,6,7).forward()
    print(res)