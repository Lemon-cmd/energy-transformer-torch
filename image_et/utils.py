import torch, numpy as np
from functools import partial
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


CIFAR10_STD = (0.4914, 0.4822, 0.4465)
CIFAR10_MU = (0.2023, 0.1994, 0.2010)

CIFAR100_STD = (0.5071, 0.4867, 0.4408)
CIFAR100_MU = (0.2675, 0.2565, 0.2761)


def gen_mask_id(num_patch, mask_size, batch_size: int):
    batch_id = torch.arange(batch_size)[:, None]
    mask_id = torch.randn(batch_size, num_patch).argsort(-1)[:, :mask_size]
    return batch_id, mask_id


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unnormalize(x, std, mean):
    x = x * std + mean
    return x


def device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    return False


def GetCIFAR(root, which: str = "cifar10"):
    which = which.lower()
    if which == "cifar10":
        std, mean = CIFAR10_STD, CIFAR10_MU

        trainset = CIFAR10(
            root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                ]
            ),
        )

        testset = CIFAR10(
            root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                ]
            ),
        )

    elif which == "cifar100":
        std, mean = CIFAR100_STD, CIFAR100_MU

        trainset = CIFAR100(
            root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                ]
            ),
        )

        testset = CIFAR100(
            root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(std, mean),
                ]
            ),
        )
    else:
        raise NotImplementedError("Not Available.")

    std, mean = map(lambda z: np.array(z)[None, :, None, None], (std, mean))

    return (trainset, testset, partial(unnormalize, std=std, mean=mean))
