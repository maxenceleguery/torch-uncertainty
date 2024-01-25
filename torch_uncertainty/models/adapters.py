import torch
from torch import nn

from torch_uncertainty.layers.normalization import BatchNormAdapter


class _Adapters(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.01,
    ) -> None:
        """Create a ABNN from a model."""
        super().__init__()

        self.model = model
        self.alpha = alpha

        # Count number of BatchNorm2d before recursively replace them
        batch_norms = [
            k.split(".")
            for k, m in model.named_modules()
            if "BatchNorm2d" in type(m).__name__
        ]

        if len(batch_norms) == 0:
            raise ValueError(
                "the model does not contain any batchNorm2d layer."
            )

        if alpha < 0:
            raise ValueError(f"alpha must be positive. Got {alpha}.")

        self.replace_bn(model, "model")

    def replace_bn(self, module, name):
        """Recursively replace BatchNorm2d with BatchNormAdapter."""
        # go through all attributes of current module and replace all BatchNorm2d
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.BatchNorm2d:
                new_batch_norm = BatchNormAdapter(
                    target_attr.num_features, alpha=self.alpha
                )
                setattr(module, attr_str, new_batch_norm)

        # iterate through immediate child modules.
        for name, immediate_child_module in module.named_children():
            self.replace_bn(immediate_child_module, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def a_bnn(model: nn.Module, alpha: float = 0.01) -> _Adapters:
    return _Adapters(model=model, alpha=alpha)
