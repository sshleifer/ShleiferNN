import numpy as np

np.random.seed(seed=42)
def numerical_gradient_old(func, inputs, h=.0001):
    numerator = func(inputs + h) - func(inputs - h)
    print numerator.shape
    denominator = 2 * h
    return numerator / denominator


def numerical_gradient(func, inputs, h=.0001):

    output = inputs.copy()
    print(output.shape)
    denominator = 2 * h
    it = np.nditer(output)
    while not it.finished:
        print("%d " % (it))
        print it.iternext()
        tmp = inputs.copy()
        tmp[it.index] = tmp[it.index] + h
        up_val = func(tmp)
        tmp[it.index] = tmp[it.index] -h
        down_val = func(tmp)
        output[it.index] = ((up_val+down_val)/denominator)
    return output


def MSE(y, yhat):
    assert y.shape == yhat.shape
    sq_error = (y - yhat) ** 2
    return np.mean(sq_error)


