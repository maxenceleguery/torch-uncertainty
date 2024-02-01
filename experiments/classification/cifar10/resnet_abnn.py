import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.adapters import a_bnn
from torch_uncertainty.models.resnet import resnet50
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet50
from torch_uncertainty.routines.classification import ClassificationEnsemble

if __name__ == "__main__":
    pl.seed_everything(0, workers=True)

    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Training")
    parser.add_argument("--lr", default=0.005, type=float, help="learning_rate")
    parser.add_argument(
        "--randomprior", default=7, type=int, help="Random prior"
    )
    parser.add_argument(
        "--alpha",
        default=0.01,
        type=float,
        help="Change alpha parameters in BatchNormAdapter2d",
    )
    parser.add_argument(
        "--onlyTrainBN",
        action="store_true",
        help="Only train BatchNorm parameters",
    )
    parser.add_argument(
        "--save_best_checkpoint",
        action="store_true",
        help="Save the best checkpoint",
    )
    args0 = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available : {use_cuda}")

    batch_size = 128
    num_classes = 10
    num_epochs = 2
    best_acc = 0

    nb_ensemble = 3

    criterion = nn.CrossEntropyLoss()
    if args0.randomprior > 1:
        weight = torch.ones([num_classes])
        ind = torch.randint(0, num_classes - 1, (1,)).item()
        ind2 = torch.randint(0, num_classes - 1, (1,)).item()
        weight[ind2] = args0.randomprior
        weight[ind] = args0.randomprior
        criterion = nn.CrossEntropyLoss(weight=weight.cuda())

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
                strict=False,
            )
        }
        net.load_state_dict(weights, strict=True)

    net = a_bnn(nets, alpha=args0.alpha)

    args = init_args(ClassificationEnsemble, CIFAR10DataModule)

    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))
    model = ClassificationEnsemble(
        num_classes=dm.num_classes,
        model=net,
        in_channels=dm.num_channels,
        loss=criterion,
        optimization_procedure=optim_cifar10_resnet50,
        baseline_type="ensemble",
        **vars(args),
    )

    cli_main(model, dm, root, "logs/abnn", args)
