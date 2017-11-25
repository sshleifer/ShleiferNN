import numpy as np
import torch
from torch.autograd import Variable
from torch import FloatTensor
from tqdm import tqdm
from torch.nn import Module

from nn.utils import MSE, numerical_gradient
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

class LinReg(object):

    def __init__(self, learning_rate=.01, n_iters=10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.verbose = False

    def fit(self, X, Y, n_batches=6):
        self.W = np.zeros((X.shape[1],))
        self.bias = 0
        self.loss_path = []
        split_X = np.array_split(X, n_batches)
        split_Y = np.array_split(Y, n_batches)
        for _ in range(self.n_iters):
            for x, y in zip(split_X, split_Y):  # TODO(batches)

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


class TorchReg(object):
    def __init__(self, learning_rate=.01, n_iters=10, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose

    def fit(self, X, y, iters=500):
        n_features = X.data.shape[1]
        n_out = y.data.shape[1]
        self.loss_path = []
        self.coeff_path = []
        self.W = Variable(torch.zeros(n_features, n_out).type(FloatTensor),
                          requires_grad=True)
        self.iter = 0
        for iter in tqdm(range(iters), desc='Iters'):
            self.iter = iter
            yhat = self.forward(X)
            self.loss  = self.calc_loss(y, yhat)
            self.loss.backward()
            update = -(self.learning_rate * self.W.grad.data)
            if self.iter < 5:
                print(update)
            self.W.data += update
            self.coeff_path.append(self.W.data.numpy()[:,0])
            self.W.grad.data.zero_()
        return self

    def forward(self, X):
        return X.mm(self.W)  # self.forward(X)

    def backward(self, y, yhat):
        raise NotImplementedError

    def calc_loss(self, y, yhat):
        '''MSE'''
        loss = (yhat - y).pow(2).sum()
        self.loss_path.append(loss)
        return loss


class TorchLogreg(TorchReg):

    def forward(self, X):
        return torch.sigmoid(X.mm(self.W))

    def calc_loss(self, y, yhat):
        loss = (torch.sum(y.mul(torch.log(yhat))))
        loss = (- loss / yhat.data.shape[0])
        self.loss_path.append(loss.data.numpy())
        return loss
        #+ (1 - actual) * log(1 - predicted))/
        # raise NotImplementedError


class TorchNN(TorchReg):

    def __init__(self, learning_rate=1e-3, n_hidden=1):
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden

    def fit(self, X, y, iters=500):
        n_features = X.data.shape[1]

        n_out = y.data.shape[1]
        self.loss_path = []
        self.coeff_path = []
        self.W1 = Variable(torch.randn(n_features, self.n_hidden).type(FloatTensor),
                           requires_grad=True)
        self.W2 = Variable(torch.randn(self.n_hidden, n_out).type(FloatTensor),
                           requires_grad=True)
        for iter in tqdm(range(iters), desc='Iters'):
            self.iter = iter
            yhat = self.forward(X)
            self.loss  = self.calc_loss(y, yhat)
            self.loss.backward()
            update1 = -(self.learning_rate * self.W1.grad.data)
            update2 = -(self.learning_rate * self.W2.grad.data)
            if self.iter < 5:
                print(update1, update2)
            self.W1.data += update1
            self.W2.data += update2
            # self.coeff_path.append(self.W1.data.numpy()[:,0])
            self.W1.grad.data.zero_()
            self.W2.grad.data.zero_()
        return self

    def forward(self, X):
        x = F.relu(X.mm(self.W1))  # self.forward(X)
        return x.mm(self.W2)


class Conv2D(Module):
    def __init__(self, filter_size=3, stride=0, padding=False):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x[::self.filter_size, ::self.filter_size]



class TorchConv(TorchReg):
    def __init__(self, learning_rate=1e-3, n_hidden=1):
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden



def convolve(a,b):
    res = []
    brev = list(reversed(b))
    for result_idx in range(len(a) + len(b) -1):
        res.append(0)
        for a_idx in range(len(a)):
            if (result_idx - a_idx + 1) > 0:
                print(result_idx - a_idx + 1)
                val = (a[a_idx] * brev[result_idx-a_idx+1])
                print(val)
                res[result_idx] += val
    return res