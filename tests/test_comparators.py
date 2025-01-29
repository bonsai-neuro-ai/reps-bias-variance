import unittest
from functools import partial
from typing import Callable

import numpy.testing as npt
import torch

from comparators import (
    prep_reps,
    double_center,
    cka,
    procrustes,
    bures,
    rsa_cosine,
)

_PRECISION = 6


def _dummy_reps(m, nx, ny):
    x = torch.randn(m, nx) + torch.abs(torch.randn(1, nx))
    y = torch.randn(m, ny) + torch.abs(torch.randn(1, ny))
    # For the purposes of testing, we'll use double precision. Single precision will then be close
    # enough for actual calculations
    return x.double(), y.double()


class TestPrepReps(unittest.TestCase):
    def test_no_center_no_scale(self):
        x, y = _dummy_reps(m=100, nx=10, ny=10)
        x2, y2 = prep_reps(x, y, center=False, scale=False)

        # Assert x and y are unchanged
        npt.assert_array_equal(x, x2)
        npt.assert_array_equal(y, y2)

        # Assert they have different memory storage
        x[0, 0] = y[0, 0] = 10000000.0
        npt.assert_raises(AssertionError, npt.assert_array_equal, x, x2)
        npt.assert_raises(AssertionError, npt.assert_array_equal, y, y2)

    def test_center_no_scale(self):
        x, y = _dummy_reps(m=100, nx=10, ny=10)
        x2, y2 = prep_reps(x, y, center=True, scale=False)

        # Assert x and y do not have mean zero
        npt.assert_raises(
            AssertionError, npt.assert_array_almost_equal, x.mean(dim=0), torch.zeros(10)
        )
        npt.assert_raises(
            AssertionError, npt.assert_array_almost_equal, y.mean(dim=0), torch.zeros(10)
        )

        # Assert x2 and y2 *do* have mean zero
        npt.assert_array_almost_equal(x2.mean(dim=0), torch.zeros(10), decimal=_PRECISION)
        npt.assert_array_almost_equal(y2.mean(dim=0), torch.zeros(10), decimal=_PRECISION)

    def test_center_m_equals_n(self):
        # Checking the 'keepdim' aspect of centering
        x, y = _dummy_reps(m=10, nx=10, ny=10)
        x2, y2 = prep_reps(x, y, center=True, scale=False)

        # Assert x2 and y2 have mean zero *along axis 0*
        npt.assert_array_almost_equal(x2.mean(dim=0), torch.zeros(10), decimal=_PRECISION)
        npt.assert_array_almost_equal(y2.mean(dim=0), torch.zeros(10), decimal=_PRECISION)

    def test_no_center_scale(self):
        x, y = _dummy_reps(m=100, nx=10, ny=10)
        x2, y2 = prep_reps(x, y, center=False, scale=True)

        # Assert there was a change
        npt.assert_raises(AssertionError, npt.assert_array_almost_equal, x, x2, decimal=_PRECISION)
        npt.assert_raises(AssertionError, npt.assert_array_almost_equal, y, y2, decimal=_PRECISION)

        # Assert the new reps have frobenius norm 1
        npt.assert_almost_equal(torch.linalg.norm(x2, ord="fro"), 1.0)
        npt.assert_almost_equal(torch.linalg.norm(y2, ord="fro"), 1.0)

    def test_center_scale(self):
        x, y = _dummy_reps(m=100, nx=10, ny=10)
        x2, y2 = prep_reps(x, y, center=True, scale=True)

        # Assert there was a change
        npt.assert_raises(AssertionError, npt.assert_array_equal, x, x2)
        npt.assert_raises(AssertionError, npt.assert_array_equal, y, y2)

        # Assert axis zero has mean 0
        npt.assert_array_almost_equal(x2.mean(dim=0), torch.zeros(10), decimal=_PRECISION)
        npt.assert_array_almost_equal(y2.mean(dim=0), torch.zeros(10), decimal=_PRECISION)

        # Assert the new reps have frobenius norm 1
        npt.assert_almost_equal(torch.linalg.norm(x2, ord="fro"), 1.0, decimal=_PRECISION)
        npt.assert_almost_equal(torch.linalg.norm(y2, ord="fro"), 1.0, decimal=_PRECISION)


class TestDoubleCenter(unittest.TestCase):
    def test_linear_kernel(self):
        x, _ = _dummy_reps(m=100, nx=10, ny=10)

        gram_x = torch.einsum("in,jn->ij", x, x)
        centered_gram_x = double_center(gram_x)

        centered_x = x - x.mean(dim=0, keepdim=True)
        gram_centered_x = torch.einsum("in,jn->ij", centered_x, centered_x)

        npt.assert_array_almost_equal(centered_gram_x, gram_centered_x, decimal=_PRECISION)


class TestMetricRelations(unittest.TestCase):

    def _test_metrics_agree_helper(self, metric1: Callable, metric2: Callable):
        x, y = _dummy_reps(m=100, nx=10, ny=10)
        similarity1 = metric1(x, y)
        similarity2 = metric2(x, y)

        # Assert a nonzero result so that things must be 'interesting'

        # Assert the result is nonzero to keep it interesting
        npt.assert_raises(
            AssertionError, npt.assert_almost_equal, similarity1, 0.0, decimal=_PRECISION
        )
        npt.assert_raises(
            AssertionError, npt.assert_almost_equal, similarity2, 0.0, decimal=_PRECISION
        )

        # Assert the two methods agree
        npt.assert_almost_equal(similarity1, similarity2, decimal=_PRECISION)

    def test_procrustes_bures_match(self):
        self._test_metrics_agree_helper(
            metric1=procrustes,
            metric2=bures,
        )

    def test_rsa_brownian_cka_match_corrected(self):
        self._test_metrics_agree_helper(
            metric1=partial(rsa_cosine, q=1.0, center=True),
            metric2=partial(cka, debias="none", kernel="brownian"),
        )

    def test_rsa_linear_cka_match_corrected(self):
        self._test_metrics_agree_helper(
            metric1=partial(rsa_cosine, q=2.0, center=True),
            metric2=partial(cka, debias="none", kernel="linear"),
        )
