import numpy as np
import unittest

from code.utils import LinReg


class TestLinReg(unittest.TestCase):

    def test_lin_reg(self):
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)
        ideal_W = [2, 3, 0]
        ideal_bias = 0
        clf = LinReg(n_iters=3)
        clf.fit(X, y)
        self.assertGreater(clf.loss_path[0], clf.loss_path[-1])
        self.assertGreater(clf.W[0], 1.)
        self.assertGreater(clf.W[1], 2.)
        coeff_error = (clf.W - ideal_W)
        self.assertGreater(.5, np.median(np.abs(coeff_error)))

