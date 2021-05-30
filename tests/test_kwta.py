import numpy as np
import torch
import unittest
from numpy.testing import assert_array_equal

from kwta import kWTA, iWTA, kWTA_different_k
from nn.kwta import KWTAFunction, IterativeWTA


class TestKWTA(unittest.TestCase):
    def test_kwta_zero(self):
        x = np.random.randint(low=0, high=100, size=100)
        x_kwta = kWTA(x, k=0)
        assert_array_equal(x_kwta, 0)

    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)

    def test_kwta_tensor(self):
        n_neurons, n_samples = 200, 10
        k = 20
        x = np.random.choice(5000, size=(n_neurons, n_samples), replace=False)
        y_array = kWTA(x, k=k)
        y_tensor = KWTAFunction.apply(torch.from_numpy(x.T), k).T
        assert_array_equal(y_array, y_tensor.squeeze())
        self.assertEqual(y_array.shape, (n_neurons, n_samples))
        assert_array_equal(y_array.sum(axis=0), k)
        y1d = np.zeros_like(y_array)
        for i, xi in enumerate(x.T):
            y1d[:, i] = kWTA(xi, k=k)
        assert_array_equal(y_array, y1d)

    def test_kwta_different_k(self):
        n_samples = 20
        k = 10
        x = np.random.choice(5000, size=(100, n_samples), replace=False)
        y1 = kWTA(x, k=k)
        y1_tensor = KWTAFunction.apply(torch.from_numpy(x.T), k).T
        assert_array_equal(y1, y1_tensor)
        ks = np.full(n_samples, fill_value=k, dtype=int)
        y2 = kWTA_different_k(x, ks=ks)
        y2_tensor = KWTAFunction.apply(torch.from_numpy(x.T), torch.from_numpy(ks)).T
        assert_array_equal(y1, y2)
        assert_array_equal(y2, y2_tensor)

    def test_kwtai_tensor(self):
        n_neurons, n_samples = 200, 10
        x = np.random.binomial(1, p=0.5, size=(n_neurons, n_samples)).astype(np.int32)
        w_xy = np.random.binomial(1, p=0.1, size=(n_neurons, n_neurons)).astype(np.int32)
        w_xh = np.random.binomial(1, p=0.1, size=(n_neurons, n_neurons)).astype(np.int32)
        y0 = w_xy @ x
        h0 = w_xh @ x
        w_lat = np.random.binomial(1, p=0.1, size=(n_neurons, n_neurons)).astype(np.int32)
        iwta = IterativeWTA(w_xy=torch.from_numpy(w_xy.T),
                            w_xh=torch.from_numpy(w_xh.T),
                            w_hy=torch.from_numpy(w_lat.T))
        h_array, y_array = iWTA(y0=y0, h0=h0, w_hy=w_lat)
        h_tensor, y_tensor = iwta(torch.from_numpy(x.T))
        assert_array_equal(h_array, h_tensor.T)
        assert_array_equal(y_array, y_tensor.T)
        self.assertEqual(y_array.shape, (n_neurons, n_samples))
        self.assertEqual(h_array.shape, (n_neurons, n_samples))
        self.assertTrue((y_array.sum(axis=0) >= 1).all())
        h1d = np.zeros_like(h_array)
        y1d = np.zeros_like(y_array)
        for i in range(n_samples):
            h1d[:, i], y1d[:, i] = iWTA(y0=y0[:, i], h0=h0[:, i], w_hy=w_lat)
        assert_array_equal(h_array, h1d)
        assert_array_equal(y_array, y1d)

    def test_kwtai_differs(self):
        y0 = np.array([0, 1, 0, 0, 0])
        h0 = np.array([0, 0, 1, 0, 0])
        w_hy = [[0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        h, y = iWTA(y0=y0, h0=h0, w_hy=w_hy)
        y_kwta = kWTA(y0 - w_hy @ h0, k=1)
        print(f"{y=}, {y_kwta=}")


if __name__ == '__main__':
    unittest.main()
