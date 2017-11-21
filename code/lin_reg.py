import numpy as np

from code.utils import MSE, numerical_gradient


class LinReg(object):

    def __init__(self, learning_rate=.01, n_iters=10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, Y, n_batches=6):
        self.W = np.zeros((X.shape[1],))
        self.bias = 0
        self.loss_path = []
        split_X = np.array_split(X, n_batches)
        split_Y = np.array_split(Y, n_batches)
        for _ in range(self.n_iters):
            for x, y in zip(split_X, split_Y):  # TODO(batches)
                # FORWARDu

                yhat = self.forward(x)
                assert yhat.shape == y.shape
                # loss = mean_squared_error(y, yhat)
                # BACKWARD
                batch_loss = lambda yhat: MSE(y, yhat)
                dyhat = numerical_gradient(
                    batch_loss, yhat)
                partial_forward = lambda W: self.forward(x, W)
                dW_dyhat = numerical_gradient(partial_forward, self.W)
                dW = dW_dyhat * dyhat  # chain rule
                # d_W = dyhat * x   # where row is derivative of output WRT weights
                d_bias = dyhat
                bias_update = d_bias * self.learning_rate
                W_update = dW * self.learning_rate
                self.W = self.W - W_update
                self.bias = self.bias - bias_update

                print('updated: W: {}: bias: {}'.format(self.W, self.bias))
            total_loss = MSE(Y, self.forward(X))
            self.loss_path.append(total_loss)

    def forward(self, X, W=None):
        if W is None:
            W = self.W
        return X.dot(W) + self.bias
