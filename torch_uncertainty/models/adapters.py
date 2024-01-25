import torch
from torch import nn


class _Adapters(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        """Create a ABNN from a model."""
        super().__init__()

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
