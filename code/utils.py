import numpy as np

np.random.seed(seed=42)
def numerical_gradient(func, inputs, h=.0001):
    numerator = func(inputs + h) - func(inputs - h)
    denominator = 2 * h
    return numerator / denominator

def mean_squared_error(y, yhat):
    assert y.shape == yhat.shape
    sq_error = (y - yhat) ** 2
    return np.mean(sq_error)


class LinReg(object):

    def __init__(self, learning_rate=.01, n_iters=10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, Y):
        self.W = np.zeros((X.shape[1],))
        self.bias = 0
        self.loss_path = []

        for iter in range(self.n_iters):
            for x, y in zip(X, Y):  # TODO(batches)
                # FORWARD
                yhat = self.forward(x)
                # loss = mean_squared_error(y, yhat)
                # BACKWARD
                batch_loss = lambda yhat: mean_squared_error(y, yhat)
                d_output = numerical_gradient(
                    batch_loss, yhat)
                d_W = d_output * x   # where row is derivative of output WRT weights
                d_bias = d_output
                bias_update = d_bias * self.learning_rate
                W_update =  d_W * self.learning_rate
                self.W = self.W - W_update
                self.bias = self.bias - bias_update

                print('updated: W: {}: bias: {}'.format(self.W, self.bias))
            total_loss = mean_squared_error(Y, self.forward(X))
            self.loss_path.append(total_loss)

    def forward(self, X):
        return X.dot(self.W) + self.bias