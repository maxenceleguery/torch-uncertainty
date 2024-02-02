from pathlib import Path

import pytorch_lightning as pl
import torch
from cli_test_helpers import ArgvContext
from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.adapters import a_bnn
from torch_uncertainty.models.resnet import resnet50
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_abnn_finetuning,
)
from torch_uncertainty.routines.classification import ClassificationEnsemble

if __name__ == "__main__":
    pl.seed_everything(0, workers=True)

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available : {use_cuda}")
    device = "gpu" if use_cuda else "cpu"

    with ArgvContext(
        "file.py",
        "--accelerator",
        device,
        "--device",
        "1",
    ):
        num_epochs = 2
        batch_size = 64
        nb_ensemble = 1
        num_samples = 1

        root = Path(__file__).parent.absolute().parents[2]

        nets = [
            resnet50(in_channels=3, num_classes=10, style="Cifar")
            for _ in range(nb_ensemble)
        ]

        checkpoint = torch.load(
            str(root / "data/resnet50_c10.ckpt")
        )  # Will be on HuggingFace later

        for net in nets:
            weights = {
                key: val
                for (_, val), (key, _) in zip(
                    checkpoint["state_dict"].items(),
                    net.state_dict().items(),
                    strict=True,
                )
            }
            net.load_state_dict(weights, strict=True)

        net = a_bnn(nets, alpha=0.01, num_samples=num_samples)

        args = init_args(ClassificationEnsemble, CIFAR10DataModule)

        args.seed = 0
        args.root = str(root / "data")
        args.batch_size = batch_size
        args.max_epochs = num_epochs
        args.num_estimators = nb_ensemble * num_samples

        args.eval_ood = True

        dm = CIFAR10DataModule(**vars(args))

        """
        criterion = nn.CrossEntropyLoss()
        if args0.randomprior > 1:
            weight = torch.ones([dm.num_classes])
            ind = torch.randint(0, dm.num_classes - 1, (1,)).item()
            ind2 = torch.randint(0, dm.num_classes - 1, (1,)).item()
            weight[ind2] = args0.randomprior
            weight[ind] = args0.randomprior
            criterion = nn.CrossEntropyLoss(weight=weight)
        """

        model = ClassificationEnsemble(
            num_classes=dm.num_classes,
            model=net,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_abnn_finetuning,
            baseline_type="ensemble",
            **vars(args),
        )

        cli_main(model, dm, root, "logs/abnn", args)
