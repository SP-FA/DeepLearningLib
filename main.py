from function import Square, Exp, Add, Config, no_grad
from tensor import Tensor


def add(x0, x1):
    return Add()(x0, x1)


def square(x):
    return Square()(x)


if __name__ == "__main__":
    with no_grad():
        x = Tensor(2)
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        print(x.grad)
