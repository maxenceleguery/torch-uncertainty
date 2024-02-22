import torch
from torch import nn

from torch_uncertainty.layers.normalization import BatchNormAdapter


class _Adapters(nn.Module):
    def __init__(
        self,
        models: list[nn.Module] | nn.Module,
        alpha: float = 0.01,
        num_samples: int = 1,
    ) -> None:
        """Create a ABNN from a model or ABNN ensemble from a list of model."""
        super().__init__()

        if isinstance(models, nn.Module):
            self.models = nn.ModuleList([models])
        elif all(isinstance(x, nn.Module) for x in models):
            self.models = nn.ModuleList(models)
        else:
            raise ValueError(
                f"Must give nn.Module or a list of nn.Module. Got {type(models)}."
            )

        self.alpha = alpha
        self.num_samples = num_samples

        if alpha < 0:
            raise ValueError(f"alpha must be positive. Got {alpha}.")

        for _model in self.models:
            # Count number of BatchNorm2d before recursively replace them
            batch_norms = [
                k.split(".")
                for k, m in _model.named_modules()
                if "BatchNorm" in type(m).__name__
            ]

            # """
            if len(batch_norms) == 0:
                raise ValueError(
                    "one model does not contain any batchNorm2d layer."
                )
            # """

            state_dict = _model.state_dict()
            self.replace_bn(_model, "model")
            _model.load_state_dict(state_dict=state_dict)

    def replace_bn(self, module, name):
        """Recursively replace BatchNorm2d with BatchNormAdapter."""
        # go through all attributes of current module and replace all BatchNorm2d
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, torch.nn.BatchNorm2d):
                new_batch_norm = BatchNormAdapter(
                    target_attr.num_features, alpha=self.alpha
                )
                setattr(module, attr_str, new_batch_norm)

        # iterate through immediate child modules.
        for name, immediate_child_module in module.named_children():
            self.replace_bn(immediate_child_module, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.models.training:  # No multi-samples during training
            return nn.functional.softmax(
                torch.cat(
                    [
                        nn.functional.softmax(model(x), dim=-1)
                        for model in self.models
                    ],
                    dim=0,
                ),
                dim=-1,
            )
        return nn.functional.softmax(
            torch.cat(
                [
                    torch.stack(
                        [
                            nn.functional.softmax(model(x), dim=-1)
                            for _ in range(self.num_samples)
                        ],
                        dim=0,
                    ).mean(dim=0)
                    for model in self.models
                ],
                dim=0,
            ),
            dim=-1,
        )


def a_bnn(
    models: nn.Module, alpha: float = 0.01, num_samples: int = 1
) -> _Adapters:
    return _Adapters(models=models, alpha=alpha, num_samples=num_samples)
