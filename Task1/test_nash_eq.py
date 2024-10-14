import unittest
import numpy as np
from nash_eq import nash_equilibrium


class TestNashEquilibrium(unittest.TestCase):

    def test_inv_inp_val(self):
        with self.assertRaises(ValueError):
            nash_equilibrium([[1, 2], [3]])

    def test_empty_m(self):
        with self.assertRaises(ValueError):
            nash_equilibrium([])

    def test_zero_str(self):
        with self.assertRaises(ValueError):
            nash_equilibrium([[]])

    def test_c1(self):
        m = [[7, 10, 4, 1],
             [6, 8, 5, 12],
             [8, -3, 2, 11]]
        exp_val = 5
        exp_p = [0, 1, 0]
        exp_q = [0, 0, 1, 0]
        p, q, val = nash_equilibrium(m)
        np.testing.assert_allclose(p, exp_p, atol=1e-2)
        np.testing.assert_allclose(q, exp_q, atol=1e-2)
        self.assertAlmostEqual(val, exp_val, places=2)

    def test_c2(self):
        m = [[4, 1, 3],
             [5, 6, 10],
             [7, 3, 4]]
        exp_val = 5.4
        exp_p = [0.0, 0.8, 0.2]
        exp_q = [0.6, 0.4, 0.0]
        p, q, val = nash_equilibrium(m)
        np.testing.assert_allclose(p, exp_p, atol=1e-2)
        np.testing.assert_allclose(q, exp_q, atol=1e-2)
        self.assertAlmostEqual(val, exp_val, places=2)

    def test_c3(self):
        m = [[8, 3, 5],
             [5, 5, 3],
             [3, 5, 5]]
        p, q, val = nash_equilibrium(m)
        exp_val = round(val, 2)
        exp_p = p
        exp_q = q
        self.assertAlmostEqual(np.sum(p), 1, places=5)
        self.assertAlmostEqual(np.sum(q), 1, places=5)
        self.assertTrue(np.all(p >= 0) and np.all(p <= 1))
        self.assertTrue(np.all(q >= 0) and np.all(q <= 1))
        p_test, q_test, val_test = nash_equilibrium(m)
        np.testing.assert_allclose(p_test, exp_p, atol=1e-2)
        np.testing.assert_allclose(q_test, exp_q, atol=1e-2)
        self.assertAlmostEqual(val_test, exp_val, places=2)


if __name__ == '__main__':
    unittest.main()
