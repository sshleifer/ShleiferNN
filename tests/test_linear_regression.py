import numpy as np
import unittest

from code.lin_reg import LinReg
from code.utils import numerical_gradient
from numpy.testing import assert_array_almost_equal

class TestLinReg(unittest.TestCase):
    def _clf_checker(self, clf, ideal_W):
        self.assertGreater(clf.loss_path[0], clf.loss_path[-1])
        self.assertGreater(clf.W[0], 1.)
        self.assertGreater(clf.W[1], 2.)
        coeff_error = (clf.W - ideal_W)
        self.assertGreater(.5, np.median(np.abs(coeff_error)))
        self.assertGreater(.2, np.median(np.abs(clf.bias)))

    def test_lin_reg(self):
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)
        ideal_W = [2, 3, 0]
        ideal_bias = 0
        clf = LinReg(n_iters=2)
        clf.fit(X, y)
        self._clf_checker(clf, ideal_W)
        clf.fit(X, y, n_batches=4)


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