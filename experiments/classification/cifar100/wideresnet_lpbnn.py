from functools import partial
from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import WideResNet
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.losses import LpbnnLoss
from torch_uncertainty.optimization_procedures import get_procedure

if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[2]

    args = init_args(WideResNet, CIFAR100DataModule)

    net_name = f"{args.version}-wideresnet28x10-cifar100"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR100DataModule(**vars(args))

    base_model = WideResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=get_procedure(
            "wideresnet28x10", "cifar100", args.version
        ),
        style="cifar",
        **vars(args),
    )

    # model
    model = WideResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=partial(
            LpbnnLoss,
            criterion=nn.CrossEntropyLoss(),
            kl_weight=1,
            model=base_model.model,
        ),
        optimization_procedure=get_procedure(
            "wideresnet28x10", "cifar100", args.version
        ),
        style="cifar",
        **vars(args),
    )
    model.model = base_model.model

    cli_main(model, dm, root, net_name, args)
