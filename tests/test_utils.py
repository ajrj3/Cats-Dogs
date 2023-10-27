import numpy as np

from catsdogs import utils


def test_set_seed():
    utils.set_seed()
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 3)
    utils.set_seed()
    c = np.random.randn(3, 3)
    d = np.random.randn(3, 3)
    assert np.array_equal(a,c)
    assert np.array_equal(b,d)