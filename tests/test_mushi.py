#! /usr/bin/env python

import unittest
import numpy as np
from mushi import histories
from mushi import utils


class TestMushi(unittest.TestCase):
    def test_constant_history(self):
        u"""test expected SFS under constant demography and mutation rate
        against the analytic formula from Fu (1995)
        """
        n = 198
        η0 = 3e4
        μ0 = 40
        change_points = np.array([])
        η = histories.eta(change_points, np.array([η0]))
        t, y = η.arrays()
        μ = histories.mu(change_points, np.array([[μ0]]))

        ξ_mushi = np.squeeze(utils.C(n) @ utils.M(n, t, y) @ μ.Z)
        ξ_Fu = 2 * η0 * μ0 / np.arange(1, n)

        self.assertTrue(np.isclose(ξ_mushi, ξ_Fu).all(),
                        msg=f'\nξ_mushi:\n{ξ_mushi}\nξ_Fu:\n{ξ_Fu}')


if __name__ == '__main__':
    unittest.main()
