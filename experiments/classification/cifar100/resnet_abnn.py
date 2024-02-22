import shutil
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from cli_test_helpers import ArgvContext
from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.models import deep_ensembles
from torch_uncertainty.models.adapters import a_bnn
from torch_uncertainty.models.resnet import resnet50
from torch_uncertainty.optimization_procedures import (
    optim_cifar100_resnet50_abnn_finetuning,
)
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from torch_uncertainty.transforms.batch import RepeatTarget

if __name__ == "__main__":
    pl.seed_everything(0, workers=True)

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available : {use_cuda}")
    device = "gpu" if use_cuda else "cpu"

    torch.set_float32_matmul_precision("medium")

    with ArgvContext(
        "file.py",
        "--accelerator",
        device,
        "--device",
        "0,",
    ):
        num_epochs = 10
        batch_size = 64
        num_emsembles = 3
        num_samples = 3
        randomprior = 7

        args = init_args(ClassificationSingle, CIFAR100DataModule)
        root = Path(__file__).parent.absolute().parents[2]
        args.seed = 0
        args.root = str(root / "data")
        args.batch_size = batch_size
        args.max_epochs = num_epochs
        args.num_estimators = num_emsembles
        args.eval_ood = True

        dm = CIFAR100DataModule(**vars(args))

        def get_weights_rp():
            weights = torch.ones([dm.num_classes])
            if randomprior > 1:
                ind = torch.randint(0, dm.num_classes - 1, (1,)).item()
                ind2 = torch.randint(0, dm.num_classes - 1, (1,)).item()
                weights[ind2] = randomprior
                weights[ind] = randomprior
            return weights.to("cuda" if use_cuda else "cpu")

        nets = [
            resnet50(in_channels=3, num_classes=dm.num_classes, style="Cifar")
            for _ in range(num_emsembles)
        ]

        checkpoint = torch.load(
            str(root / "data/resnet-50_c100.ckpt")
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

        single_nets = [
            a_bnn(net, alpha=0.01, num_samples=num_samples) for net in nets
        ]

        routines = [
            ClassificationSingle(
                num_classes=dm.num_classes,
                model=net,
                in_channels=dm.num_channels,
                loss=partial(nn.CrossEntropyLoss, weight=get_weights_rp()),
                optimization_procedure=optim_cifar100_resnet50_abnn_finetuning,
                **vars(args),
            )
            for net in single_nets
        ]

        if Path.exists(Path("logs/abnn")):
            shutil.rmtree("logs/abnn")

        paths = []
        for i, routine in enumerate(routines):
            cli_main(routine, dm, root, f"logs/abnn/resnet50_c100_{i}", args)
            p = Path()
            paths.append(
                next(
                    p.glob(
                        f"logs/abnn/resnet50_c100_{i}/version_0/checkpoints/epoch*.ckpt"
                    )
                )
            )

        models = []
        for net, path in zip(single_nets, paths, strict=False):
            sd = torch.load(path)["state_dict"]
            net.load_state_dict(sd, strict=False)
            models.append(net)

        model = deep_ensembles(models)
        routine = ClassificationEnsemble(
            num_classes=dm.num_classes,
            model=model,
            loss=None,
            optimization_procedure=None,
            in_channels=dm.num_channels,
            format_batch_fn=RepeatTarget(num_repeats=num_emsembles),
            **vars(args),
        )
        trainer = pl.Trainer(False, False, accelerator="gpu", devices="0,")
        metrics = trainer.test(routine, datamodule=dm)
