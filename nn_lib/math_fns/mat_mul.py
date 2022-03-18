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
        a=self.args[0].data
        b=self.args[1].data
        #self.args[1].data*grad_output, self.args[0].data*grad_output

        return (np.matmul(b,grad_output).reshape(a.shape),np.matmul(grad_output,a).reshape(b.shape))
