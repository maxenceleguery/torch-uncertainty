import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn, optim

from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.adapters import a_bnn
from torch_uncertainty.models.resnet import resnet50

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
        "-OTBN",
        action="store_true",
        help="Only train BatchNorm parameters",
    )
    parser.add_argument(
        "--save_best_checkpoint",
        action="store_true",
        help="Save the best checkpoint",
    )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available : {use_cuda}")

    batch_size = 128
    num_classes = 10
    num_epochs = 2
    best_acc = 0

    criterion = nn.CrossEntropyLoss()
    if args.randomprior > 1:
        weight = torch.ones([num_classes])
        ind = torch.randint(0, num_classes - 1, (1,)).item()
        ind2 = torch.randint(0, num_classes - 1, (1,)).item()
        weight[ind2] = args.randomprior
        weight[ind] = args.randomprior
        criterion = nn.CrossEntropyLoss(weight=weight.cuda())

    root = Path(__file__).parent.absolute().parents[2]

    dm = CIFAR10DataModule(
        root=str(root / "data"),
        eval_ood=False,
        ood_detection=True,
        batch_size=batch_size,
    )
    dm.prepare_data()
    dm.setup()
    dm.setup("test")
    trainloader = dm.train_dataloader()
    trainset = trainloader.dataset
    testloader = dm.test_dataloader()[0]
    testset = testloader.dataset

    net = resnet50(in_channels=3, num_classes=10, style="Cifar")
    checkpoint = torch.load(
        str(root / "data/resnet50_c10.ckpt")
    )  # Will be on HuggingFace later

    weights = {
        key: val
        for (_, val), (key, _) in zip(
            checkpoint["state_dict"].items(),
            net.state_dict().items(),
            strict=False,
        )
    }

    net.load_state_dict(weights, strict=True)
    net = a_bnn(net, alpha=args.alpha)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count())
        )

    if args.onlyTrainBN:
        params_bn_tmp = list(
            filter(
                lambda kv: ("bn" in kv[0]),
                net.named_parameters(),
            )
        )
        params_bn = [param for _, param in params_bn_tmp]

        params_to_optimize = params_bn
    else:
        params_to_optimize = list(net.parameters())

    optimizer = optim.SGD(
        params_to_optimize,
        weight_decay=5e-4,
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
    )

    def train(epoch):
        net.train()
        net.training = True
        train_loss = 0
        correct = 0
        total = 0
        print(
            "\n=> Training Epoch #%d, LR=%.4f"
            % (epoch, optimizer.param_groups[0]["lr"])
        )
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            optimizer.step()  # Optimizer update
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0:
                sys.stdout.write("\r")
                sys.stdout.write(
                    "| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%% "
                    % (
                        epoch,
                        num_epochs,
                        batch_idx + 1,
                        (len(trainset) // batch_size) + 1,
                        loss.item(),
                        100.0 * correct / total,
                    )
                )
                sys.stdout.flush()

    def test(epoch):
        global best_acc
        net.eval()
        net.training = False
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            # Save checkpoint when best model
            acc = 100.0 * correct / total
            print(
                "\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%"
                % (epoch, loss.item(), acc)
            )

            if acc > best_acc:
                print("| New Best model...\t\t\tTop1 = %.2f%%" % (acc))
                if args.save_best_checkpoint:
                    state = {
                        "dict": net.state_dict(),
                        "acc": acc,
                        "epoch": epoch,
                    }
                    if not Path.isdir("checkpoint"):
                        Path.mkdir("checkpoint")
                    save_point = "./checkpoint/" + args.dataset + os.sep
                    if not Path.isdir(save_point):
                        Path.mkdir(save_point)
                    save_point = save_point + args.dirsave_out + os.sep
                    if not Path.isdir(save_point):
                        Path.mkdir(save_point)

                    torch.save(state, save_point + "resnet-50-abnn.t7")
                best_acc = acc

    for epoch in range(num_epochs):
        train(epoch)
        test(epoch)

    print("* Test results : Acc@1 = %.2f%%" % (best_acc))
