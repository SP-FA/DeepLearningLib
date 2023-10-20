import numpy as np


class Tensor:
    """实现可以自动求导的张量

    Attributes:
        data (ndarray): 张量数据
        grad (float): 张量的梯度
        generator (Function): 生成该张量的函数
    """
    def __init__(self, data):
        self.data = np.atleast_1d(data)
        self.grad = None
        self._generator = None
        self.priority = 0

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, func):
        self._generator = func
        self.priority = func.priority + 1

    def __str__(self):
        return f"data: {self.data}, shape: {self.data.shape}, grad: {self.grad}"

    def backward(self, retain_graph=False):
        """自动反向传播

        """
        if self.grad is None:
            self.grad = 1

        funcs = []

        def add_func(f):
            if f is None: return
            if f not in funcs:
                funcs.append(f)
                funcs.sort(key=lambda x: x.priority)

        add_func(self.generator)
        while len(funcs):
            generator = funcs.pop()
            inputs = generator.inputs
            outputs = generator.outputs

            gys = [o().grad for o in outputs]
            gxs = generator.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for input, gx in zip(inputs, gxs):
                if input.grad is None:
                    input.grad = gx
                else:
                    input.grad = input.grad + gx
                add_func(input.generator)

            if not retain_graph:
                for o in outputs:
                    o().grad = None

    # def zero_grad(self):
    #     self.grad = None
