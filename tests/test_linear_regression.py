import numpy as np
import torch
from torch.autograd import Variable
from torch import FloatTensor
import unittest

from nn.models import LinReg, TorchReg, TorchLogreg, TorchNN
from nn.utils import numerical_gradient
from nn.torch_testing_utils import TestCase as TorchTestCase
from numpy.testing import assert_array_almost_equal

class TestLinReg(unittest.TestCase):
    def _clf_checker(self, clf, ideal_W):
        self.assertGreater(clf.loss_path[0], clf.loss_path[-1])
        self.assertGreater(clf.W[0], 1.)
        self.assertGreater(clf.W[1], 2.)
        coeff_error = (clf.W - ideal_W)
        self.assertGreater(.5, np.median(np.abs(coeff_error)))
        self.assertGreater(.2, np.median(np.abs(clf.bias)))

    @unittest.SkipTest
    def test_lin_reg(self):
        X = Variable(torch.randn(100, 3), requires_grad=False)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)
        ideal_W = [2, 3, 0]
        ideal_bias = 0
        clf = LinReg(n_iters=2)
        clf.fit(X, y)
        self._clf_checker(clf, ideal_W)
        clf.fit(X, y, n_batches=4)

    @unittest.SkipTest
    def test_numerical_gradient(self):
        assert_array_almost_equal(
            numerical_gradient(np.sum, np.array([1,2,3])),
            np.ones(3)
        )

    @unittest.skipUnless(False, '')
    def test_numerical_gradient_matrix(self):

        X = np.random.randn(10, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)
        W = np.array([2.0, 2.0, 0])
        loss_fn = lambda w: y - X.dot(W)
        self.assertEqual(loss_fn(W).shape, y.shape)
        vec_grad = numerical_gradient(lambda w: np.sum(X.dot(w)), W)
        self.assertEqual(vec_grad.shape, X.shape)

def get_assertable_data(torch_obj):
    if isinstance(torch_obj, Variable):
        return torch_obj.data.numpy()
    elif hasattr(torch_obj, 'numpy'):
        return torch_obj.numpy()
    else:
        return torch_obj


class TestTorchReg(TorchTestCase):



    def assertGreater(self, a, b):
        adata = get_assertable_data(a)
        bdata = get_assertable_data(b)
        super(TestTorchReg, self).assertGreater(adata, bdata)

    def check_convergence(self, clf, ideal_W):
        self.assertGreater(clf.loss_path[0],
                           clf.loss_path[-1])
        #3, '{} {}'.format(clf.loss_path[0], clf.loss_path[-1]))
        coeffs = clf.W.data.numpy()[:, 0]
        coeff_error = (coeffs - ideal_W)
        self.assertGreater(.5, np.median(np.abs(coeff_error)))
        # self.assertGreater(.2, np.median(np.abs(clf.bias)))

    def test_torch_reg(self):
        n_samples = 100
        n_features = 3
        n_out = 1
        X = Variable(torch.randn(n_samples, n_features).type(FloatTensor), requires_grad=False)
        y = Variable(torch.randn(n_samples, n_out).type(FloatTensor), requires_grad=False)
        y.add_(X[:, 0] * 2)
        y.add_(X[:,1] * 3)
        torch_reg = TorchReg(learning_rate=1e-3).fit(X, y, iters=100)
        coeffs = torch_reg.W.data.numpy()[:,0]
        print(coeffs)
        self.assertGreater(coeffs[0], 1.8)
        self.assertGreater(coeffs[1], 2.8)
        self.check_convergence(torch_reg, ideal_W=[2, 3, 0])

    def test_log_reg(self):
        n_samples = 100
        n_features = 3
        n_out = 1
        X = Variable(torch.randn(n_samples, n_features).type(FloatTensor), requires_grad=False)
        y = Variable(torch.randn(n_samples, n_out).type(FloatTensor), requires_grad=False)
        y.add_(X[:, 0] * 2)
        y.add_(X[:, 1] * 3)
        y = y.ge(0).type(FloatTensor)
        torch_log_reg = TorchLogreg(learning_rate=1e-2, verbose=True).fit(X, y, iters=100)
        self.assertGreater(torch_log_reg.loss_path[0],
                           torch_log_reg.loss_path[-1])
        self.check_convergence(torch_log_reg, ideal_W=[.4, 8, 0])

    def test_nn(self):
        n_samples = 100
        n_features = 3
        n_out = 1
        X = Variable(torch.randn(n_samples, n_features).type(FloatTensor), requires_grad=False)
        y = Variable(torch.randn(n_samples, n_out).type(FloatTensor), requires_grad=False)
        y.add_(X[:, 0] * 2)
        y.add_(X[:, 1] * 3)
        torch_nn = TorchNN(learning_rate=1e-3,  n_hidden=4).fit(X, y, iters=100)
        self.assertGreater(torch_nn.loss_path[0],
                           torch_nn.loss_path[-1])
        # raise ValueError
        # self.check_convergence(torch_log_reg, ideal_W=[.4, 8, 0])

