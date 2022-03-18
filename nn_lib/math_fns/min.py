from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Min(Function):
    """
    Minimum over two arrays
    """


    def forward(self) -> np.ndarray:
        """
        Compute minimum over two arrays element-wise, i.e. result[index] =  min(a[index], b[index])

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: minimum over the two arguments
        """
        return np.minimum(self.args[0].data, self.args[1].data)

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over arguments of the minimum operation
        Important: if two values at some position are equal, the gradient is set to be 0.5

        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the minimum operation
        :return: a tuple of gradients over arguments of the minimum
        """
        #print(self.args[0].data,self.args[1].data)


        #scalars
        if (self.args[0].data.shape and self.args[1].data.shape) == ():
            if self.args[0].data==self.args[1].data:
                return (0.5*grad_output, 0.5*grad_output)
            elif self.args[0].data == np.minimum(self.args[0].data, self.args[1].data):
                return (1*grad_output,0*grad_output)
            elif self.args[1].data == np.minimum(self.args[0].data, self.args[1].data):
                return (0 * grad_output, 1 * grad_output)
        else:
            a=self.args[0].data
            b=self.args[1].data
            c=np.minimum(a,b)
            print(c)
            d1 = np.zeros(shape=(c.size, c.size))
            d2 = np.zeros(shape=(c.size, c.size))
            for i in range(len(c)):
                if (c[i] == a[i]) and (c[i] == b[i]):
                    d1[i][i]=0.5
                    d2[i][i]=0.5
                elif c[i] == a[i]:
                    d1[i][i]=1
                    d2[i][i]=0
                elif c[i] == b[i]:
                    d1[i][i] = 0
                    d2[i][i] = 1

        return (d1 * grad_output, d2 * grad_output)