# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import optim

from torch_uncertainty.models.wideresnet.batched import batched_wideresnet28x10
from torch_uncertainty.routines.classification import ClassificationEnsemble


# fmt: on
class BatchedWideResNet(ClassificationEnsemble):
    r"""LightningModule for BatchEnsembles WideResNet.

    Args:
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        in_channels (int): Number of input channels.
        loss (torch.nn.Module): Training loss.
        optimization_procedure (Any): Optimization procedure, corresponds to
            what expect the `LightningModule.configure_optimizers()
            <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
            method.
        use_entropy (bool, optional): Indicates whether to use the entropy
            values as the OOD criterion or not. Defaults to ``False``.
        use_logits (bool, optional): Indicates whether to use the logits as the
            OOD criterion or not. Defaults to ``False``.
        use_mi (bool, optional): Indicates whether to use the mutual
            information as the OOD criterion or not. Defaults to ``False``.
        use_variation_ratio (bool, optional): Indicates whether to use the
            variation ratio as the OOD criterion or not. Defaults to ``False``.

    Note:
        The OOD criterion is by defaults the confidence score.

    Warning:
        Make sure at most only one of :attr:`use_entropy`, :attr:`use_logits`,
        :attr:`use_mi` and :attr:`use_variation_ratio` attributes is set to
        ``True``. Otherwise a :class:`ValueError()` will be raised.
    """

    def __init__(
        self,
        num_classes: int,
        num_estimators: int,
        in_channels: int,
        loss: nn.Module,
        optimization_procedure: Any,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        imagenet_structure: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            num_estimators=num_estimators,
            use_entropy=use_entropy,
            use_logits=use_logits,
            use_mi=use_mi,
            use_variation_ratio=use_variation_ratio,
        )

        # construct config
        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = batched_wideresnet28x10(
            in_channels=in_channels,
            num_estimators=num_estimators,
            num_classes=num_classes,
            imagenet_structure=imagenet_structure,
        )

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 32, 32)

    def configure_optimizers(self) -> dict:
        param_optimizer = self.optimization_procedure(self)["optimizer"]
        weight_decay = param_optimizer.defaults["weight_decay"]
        lr = param_optimizer.defaults["lr"]
        momentum = param_optimizer.defaults["momentum"]
        my_list = ["R", "S"]
        params_multi_tmp = list(
            filter(
                lambda kv: (my_list[0] in kv[0]) or (my_list[1] in kv[0]),
                self.named_parameters(),
            )
        )
        param_core_tmp = list(
            filter(
                lambda kv: (my_list[0] not in kv[0])
                and (my_list[1] not in kv[0]),
                self.named_parameters(),
            )
        )
        params_multi = [param for _, param in params_multi_tmp]
        param_core = [param for _, param in param_core_tmp]
        optimizer = optim.SGD(
            [
                {"params": param_core, "weight_decay": weight_decay},
                {"params": params_multi, "weight_decay": 0.0},
            ],
            lr=lr,
            momentum=momentum,
        )
        scheduler = self.optimization_procedure(self)["lr_scheduler"]
        scheduler.optimizer = optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        input = input.repeat(self.num_estimators, 1, 1, 1)
        return self.model.forward(input)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the model's attributes via command-line options:

        - ``--num_estimators [int]``: defines :attr:`num_estimators`. Defaults
          to ``1``.
        - ``--imagenet_structure``: sets :attr:`imagenet_structure`. Defaults
          to ``True``.
        - ``--entropy``: sets :attr:`use_entropy` to ``True``.
        - ``--logits``: sets :attr:`use_logits` to ``True``.
        - ``--mutual_information``: sets :attr:`use_mi` to ``True``.
        - ``--variation_ratio``: sets :attr:`use_variation_ratio` to ``True``.

        Example:

            .. parsed-literal::

                python script.py --num_estimators 4
        """
        parent_parser.add_argument("--num_estimators", type=int, default=4)
        parent_parser.add_argument(
            "--imagenet_structure",
            action=BooleanOptionalAction,
            default=True,
            help="Use imagenet structure",
        )
        parent_parser.add_argument(
            "--entropy", dest="use_entropy", action="store_true"
        )
        parent_parser.add_argument(
            "--logits", dest="use_logits", action="store_true"
        )
        parent_parser.add_argument(
            "--mutual_information", dest="use_mi", action="store_true"
        )
        parent_parser.add_argument(
            "--variation_ratio", dest="use_variation_ratio", action="store_true"
        )
        return parent_parser