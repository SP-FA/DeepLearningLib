import weakref
from contextlib import contextmanager
import numpy as np
from tensor import Tensor


class Config:
    no_grad = False


@contextmanager
def no_grad():
    Config.no_grad = True
    try:
        yield
    except Exception:
        raise Exception("No grad exp")
    finally:
        Config.no_grad = False


class Function:
    """实现各种函数以及函数反向传播的基类

    Attributes:
        inputs (Tensor): 接收一个张量，并进行计算
    """
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        outputs = (outputs, ) if not isinstance(outputs, tuple) else outputs

        if not Config.no_grad:
            self.inputs = inputs
            self.outputs = [weakref.ref(o) for o in outputs]
            self.priority = max(input.priority for input in self.inputs)

            for y in outputs:
                y.generator = self

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        """实现函数的前向传播

        Attributes:
            inputs (Tensor): 接收一个张量，并进行计算

        Raises:
            NotImplementedError: 子类需要实现该方法，若没有实现则抛出该异常
        """
        raise NotImplementedError

    def backward(self, gy):
        """实现函数的反向传播

        Attributes:
            gy (int, float): 前一层反向传播算出的梯度

        Raises:
            NotImplementedError: 子类需要实现该方法，若没有实现则抛出该异常
        """
        raise NotImplementedError


class Square(Function):
    """实现平方函数

    """
    def forward(self, inputs):
        res = inputs.data ** 2
        return Tensor(res)

    def backward(self, gy):
        return 2 * self.inputs[0].data * gy


class Exp(Function):
    """实现 exp 函数

    """
    def forward(self, inputs):
        res = np.exp(inputs.data)
        return Tensor(res)

    def backward(self, gy):
        return np.exp(self.inputs[0].data) * gy


class Add(Function):
    def forward(self, x0, x1):
        return Tensor(x0.data + x1.data)

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        return Tensor(x0.data * x1.data)
