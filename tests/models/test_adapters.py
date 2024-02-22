import pytest
import torch

from tests._dummies import dummy_model
from torch_uncertainty.models.adapters import a_bnn
from torch_uncertainty.models.resnet import resnet50


class TestAdaptersWrapper:
    """Testing the adapters wrapper."""

    def test_main(self):
        rand = torch.randn(6, 3, 4, 4)
        resnet = resnet50(3, 10)
        model = a_bnn(resnet, alpha=0.1)
        assert model(rand).shape == (6, 10)
        assert not torch.allclose(
            model(rand), model(rand)
        ), (
            "Same output but shouldn't"
        )  # Check if there is random during inference
        # assert torch.allclose(resnet(rand),resnet(rand)), "Different output but shouldn't" # Check if there is no random during inference

    def test_main_ens(self):
        rand = torch.randn(6, 3, 4, 4)
        nb_ens = 3
        num_samples = 5
        model = a_bnn(
            [resnet50(3, 10) for _ in range(nb_ens)],
            alpha=0.1,
            num_samples=num_samples,
        )
        assert model(rand).shape == (6 * nb_ens, 10)

    def test_value_error_alpha(self):
        with pytest.raises(ValueError):
            a_bnn(resnet50(3, 10), alpha=-2.0)

    def test_no_batchnorm(self):
        with pytest.raises(ValueError):
            a_bnn(dummy_model(1, 10, 1), alpha=0.01)
