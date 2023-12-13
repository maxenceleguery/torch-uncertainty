from functools import partial
from pathlib import Path

from torch import nn, optim

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import WideResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.losses import LpbnnLoss
from torch_uncertainty.optimization_procedures import get_procedure

if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[2]

    args = init_args(WideResNet, CIFAR10DataModule)

    net_name = f"logs/{args.version}-wideresnet28x10-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    base_model = WideResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=get_procedure(
            "wideresnet28x10", "cifar10", args.version
        ),
        style="cifar",
        **vars(args),
    )

    def optim_cifar10_wideresnet(
        model: nn.Module,
    ):
        my_list = ["alpha", "gamma"]
        params_multi_tmp = list(
            filter(
                lambda kv: (my_list[0] in kv[0]) or (my_list[1] in kv[0]),
                model.named_parameters(),
            )
        )
        param_core_tmp = list(
            filter(
                lambda kv: (my_list[0] not in kv[0])
                and (my_list[1] not in kv[0]),
                model.named_parameters(),
            )
        )
        params_multi = [param for name, param in params_multi_tmp]
        param_core = [param for name, param in param_core_tmp]
        optimizer = optim.SGD(
            [
                {"params": param_core, "weight_decay": 5e-4},
                {"params": params_multi, "weight_decay": 0.0},
            ],
            lr=0.1,
            momentum=0.9,
            nesterov=True,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        optimization_procedure=optim_cifar10_wideresnet,
        style="cifar",
        **vars(args),
    )
    model.model = base_model.model

    cli_main(model, dm, root, net_name, args)
