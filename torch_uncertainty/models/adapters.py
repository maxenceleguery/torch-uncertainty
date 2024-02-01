import torch
from torch import nn

from torch_uncertainty.layers.normalization import BatchNormAdapter


class _Adapters(nn.Module):
    def __init__(
        self,
        model: list[nn.Module] | nn.Module,
        alpha: float = 0.01,
        num_samples: int = 1,
    ) -> None:
        """Create a ABNN from a model or ABNN ensemble from a list of model."""
        super().__init__()

        if isinstance(model, nn.Module):
            self.models = [model]
        else:
            self.models = model

        self.alpha = alpha
        self.num_samples = num_samples

        if alpha < 0:
            raise ValueError(f"alpha must be positive. Got {alpha}.")

        for model in self.models:
            # Count number of BatchNorm2d before recursively replace them
            batch_norms = [
                k.split(".")
                for k, m in model.named_modules()
                if "BatchNorm2d" in type(m).__name__
            ]

            if len(batch_norms) == 0:
                raise ValueError(
                    "one model does not contain any batchNorm2d layer."
                )

            state_dict = model.state_dict()
            self.replace_bn(model, "model")
            model.load_state_dict(state_dict=state_dict)

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
        if len(self.models) == 1:
            return self.models[0](x)
        return torch.cat(
            [
                model(x)
                for _ in range(self.num_samples)
                for model in self.models
            ],
            dim=0,
        )


def a_bnn(model: nn.Module, alpha: float = 0.01) -> _Adapters:
    return _Adapters(model=model, alpha=alpha)
