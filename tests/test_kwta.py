import unittest
import numpy as np
from numpy.testing import assert_array_equal

from kwta import kWTA, kWTAi, overlap, cosine_similarity


class TestKWTA(unittest.TestCase):
    def test_kwta_zero(self):
        x = np.random.randint(low=0, high=100, size=100)
        x_kwta = kWTA(x, k=0)
        assert_array_equal(x_kwta, 0)

    def test_kwta_tensor(self):
        np.random.seed(1)
        n_neurons, n_samples = 200, 10
        k = 20
        x = np.random.randint(low=0, high=100, size=(n_neurons, n_samples))
        y_tensor = kWTA(x, k=k)
        self.assertEqual(y_tensor.shape, (n_neurons, n_samples))
        assert_array_equal(y_tensor.sum(axis=0), k)
        y1d = np.zeros_like(y_tensor)
        for i, xi in enumerate(x.T):
            y1d[:, i] = kWTA(xi, k=k)
        assert_array_equal(y_tensor, y1d)

    def test_kwtai_tensor(self):
        np.random.seed(2)
        n_neurons, n_samples = 200, 10
        y0 = np.random.randint(low=0, high=100, size=(n_neurons, n_samples))
        h0 = np.random.randint(low=0, high=100, size=(n_neurons, n_samples))
        w_lat = np.random.binomial(1, p=0.1, size=(n_neurons, n_neurons))
        h_tensor, y_tensor = kWTAi(y0=y0, h0=h0, w_hy=w_lat)
        self.assertEqual(y_tensor.shape, (n_neurons, n_samples))
        self.assertEqual(h_tensor.shape, (n_neurons, n_samples))
        self.assertTrue((y_tensor.sum(axis=0) >= 1).all())
        h1d = np.zeros_like(h_tensor)
        y1d = np.zeros_like(y_tensor)
        for i in range(n_samples):
            h1d[:, i], y1d[:, i] = kWTAi(y0=y0[:, i], h0=h0[:, i], w_hy=w_lat)
        assert_array_equal(h_tensor, h1d)
        assert_array_equal(y_tensor, y1d)


if __name__ == '__main__':
    unittest.main()
