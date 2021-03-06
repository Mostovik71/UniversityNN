from typing import Tuple
import numpy as np
stop=1
from nn_lib.math_fns.function import Function


class MatMul(Function):
    """
    Matrix multiplication function
    """


    def forward(self) -> np.ndarray:
        """
        Multiply two matrices and return their product, matrices are not necessarily 2D, hint:
        https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

        :return: matrix product of the two arguments
        """
        return np.matmul(np.matrix(self.args[0].data),np.matrix(self.args[1].data))

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over two matrix multiplication arguments

        :param grad_output: gradient over the result of the multiplication operation
        :return: a tuple of gradients over two multiplication arguments
        """


        return (np.matmul(grad_output, self.args[1].data.T), np.matmul(self.args[0].data.T, grad_output))
