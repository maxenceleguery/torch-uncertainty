import pytest
import torch

from torch_uncertainty.layers.normalization import BatchNormAdapter


@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.rand((5, 20, 3, 3))


class TestBatchNormAdapters:
    def test_batch_norm_adapters(self, feat_input: torch.Tensor):
        layer = BatchNormAdapter(20, alpha=0.01)
        out = layer(feat_input)
        assert out.shape == torch.Size([5, 20, 3, 3])
