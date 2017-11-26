import numpy as np
from scipy.misc import derivative

np.random.seed(seed=42)

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    # fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        original_value = x.copy()[ix]
        x[ix] = original_value + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = original_value - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = original_value # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print (ix, grad[ix], fxph, fxmh)
        it.iternext() # step to next dimension

    return grad


def MSE(y, yhat):
    assert y.shape == yhat.shape
    sq_error = (y - yhat) ** 2
    return np.mean(sq_error)



def torch_assert_true(a, b):

    torch.le(clf.loss_path[0], clf.loss_path[-1]).equal(torch.ByteTensor([1]))
