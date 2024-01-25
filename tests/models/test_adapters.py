import pytest
import torch

from tests._dummies import dummy_model
from torch_uncertainty.models.adapters import a_bnn
from torch_uncertainty.models.resnet import resnet50


class TestAdaptersWrapper:
    """Testing the adapters wrapper."""

    def test_main(self):
        model = a_bnn(resnet50(3, 10), alpha=0.01)
        assert model(torch.randn(6, 3, 4, 4)).shape == (6, 10)

    def test_value_error_alpha(self):
        with pytest.raises(ValueError):
            a_bnn(resnet50(3, 10), alpha=-2.0)

    def test_no_batchnorm(self):
        with pytest.raises(ValueError):
            a_bnn(dummy_model(1, 10, 1), alpha=0.01)
