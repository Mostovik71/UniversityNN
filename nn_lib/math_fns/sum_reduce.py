from typing import Union, Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class SumReduce(Function):
    """
    Summation reduction over given axis (or axes)
    """

    def __init__(self, *args, axis: Union[int, Tuple[int, ...], None] = None):
        """
        Create sum reduction function

        :param args: an argument to apply sum for
        :param axis: axis or multiple axes to sum over
        """
        super(SumReduce, self).__init__(*args)#?????????
        if axis is None:
            axis = tuple(range(len(self.args[0].data.shape)))

        if isinstance(axis, int):
            axis = (axis,)

        self.axis = axis  # (0,1) if axis is None;(0,) or (1,) if axis is not None

    def forward(self) -> np.ndarray:
        # print(self.axis)
        if len(self.axis) == 2:
            return sum(sum(self.args[0].data))
        if len(self.axis) == 1:
            return sum(self.args[0].data)

        # elif self.axis==1:
        """
        Reduce given axes by summing values over them
        Hint: https://numpy.org/doc/stable/reference/generated/numpy.sum.html

        :return: the reduced value
        """

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute the gradient over the reduction operation
        Hint: values across the reduced axes must have the same gradients

        :param grad_output: the gradient of the result of the reduction
        :return: a tuple with a single value representing the gradient over the reduction argument
        """
        return (np.ones_like(self.args[0].data)*grad_output,)


