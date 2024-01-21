import os, tqdm
import torch
import argparse
import pprint
import numpy as np
import torchvision
from einops import reduce
from torch.utils.data.dataloader import DataLoader
from image_et import (
    ImageET as ET,
    Patch,
    GetCIFAR,
    gen_mask_id,
    count_parameters,
    device,
    str2bool,
)

DEFAULT_FOLDER = "./results"

parser = argparse.ArgumentParser(
    description="Train ET as Mask Auto-encoder",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--patch_size", default=4, type=int, help="patch size")

parser.add_argument(
    "--qk_dim", default=64, type=int, help="embedding dimension for attention block"
)

parser.add_argument("--mask_ratio", default=0.85, type=float, help="masking ratio")

parser.add_argument("--blocks", default=1, type=int, help="number of ET blocks")

parser.add_argument("--out_dim", default=None, type=int, help="output_dim")

parser.add_argument("--tkn_dim", default=256, type=int, help="token dimension")

parser.add_argument("--nheads", default=12, type=int, help="number of attention heads")

parser.add_argument("--attn_beta", default=None, type=float, help="attention beta")

parser.add_argument(
    "--hn_mult", default=4.0, type=float, help="hopfield multiplier value"
)

parser.add_argument(
    "--alpha", default=1.0, type=float, help="step size for ET's dynamic"
)

parser.add_argument(
    "--attn_bias",
    default=False,
    type=str2bool,
    help="a flag indicating the usage of biases in attention",
)

parser.add_argument(
    "--hn_bias",
    default=False,
    type=str2bool,
    help="a flag indicating the usage of biases in hopfield",
)

parser.add_argument(
    "--time_steps",
    default=10,
    type=int,
    help="number of timesteps for ET",
)

parser.add_argument(
    "--result_path",
    default=DEFAULT_FOLDER,
    type=str,
    help="path to save the result folder",
)

parser.add_argument(
    "--num_workers", default=0, type=int, help="number of workers for data loader"
)

parser.add_argument(
    "--batch_size", default=128, type=int, help="batch size to train the model"
)

parser.add_argument("--epochs", default=100, type=int, help="number of training points")

parser.add_argument("--learning_rate", default=8e-5, type=float, help="learning rate")

parser.add_argument("--b1", default=0.99, type=float, help="adam beta1")

parser.add_argument("--b2", default=0.999, type=float, help="adam beta2")

parser.add_argument(
    "--avg_gpu",
    default=True,
    type=str2bool,
    help="a flag indicating to divide loss by the number of devices",
)

parser.add_argument(
    "--weight_decay", default=0.001, type=float, help="weight decay value"
)

parser.add_argument(
    "--data_path", default="./", type=str, help="root folder of dataset"
)

parser.add_argument("--data_name", default="CIFAR10", type=str, help="CIFAR10/CIFAR100")

args = parser.parse_args()
config = vars(args)
pprint.pprint(config, width=1)

with open("model_config.txt", "w") as f:
    pprint.pprint(config, f, width=1)

IMAGE_FOLDER = DEFAULT_FOLDER + "/images"
MODEL_FOLDER = DEFAULT_FOLDER + "/models"

if not os.path.isdir(DEFAULT_FOLDER):
    os.mkdir(DEFAULT_FOLDER)

if not os.path.isdir(IMAGE_FOLDER):
    os.mkdir(IMAGE_FOLDER)

if not os.path.isdir(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

x = torch.randn(1, 3, 32, 32)

patch_fn = Patch(dim=config["patch_size"])

model = ET(
    x,
    patch_fn,
    config["out_dim"],
    config["tkn_dim"],
    config["qk_dim"],
    config["nheads"],
    config["hn_mult"],
    config["attn_beta"],
    config["attn_bias"],
    config["hn_bias"],
    time_steps=config["time_steps"],
    blocks=config["blocks"],
    hn_fn=lambda x: -0.5 * (torch.nn.functional.relu(x) ** 2.0).sum(),
)

print("\nPARAM. COUNT:", count_parameters(model))

NUM_PATCH = model.patch.N
NUM_MASKS = int(model.patch.N * config["mask_ratio"])

trainset, testset, unnormalize_fn = GetCIFAR(config["data_path"], config["data_name"])

train_loader, test_loader = map(
    lambda z: DataLoader(
        z,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        pin_memory=True,
    ),
    (trainset, testset),
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    betas=(config["b1"], config["b2"]),
    weight_decay=config["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    cooldown=0,
    patience=10,
    factor=0.85,
)

DEVICE = device()
model = torch.nn.DataParallel(model)
model = model.to(DEVICE)

print("\nDEVICE:", DEVICE)

NUM_DEVICE = torch.cuda.device_count()
NUM_DEVICE = 1 if NUM_DEVICE == 0 else NUM_DEVICE
NUM_DEVICE = NUM_DEVICE if config["avg_gpu"] else 1

visual_num = min(config["batch_size"], 256)
training_display = tqdm.tqdm(range(1, config["epochs"] + 1))

for i in training_display:
    running_loss = 0.0

    for x, _ in train_loader:
        mask_id = gen_mask_id(NUM_PATCH, NUM_MASKS, x.size(0))
        y = patch_fn(x)[:, mask_id]

        x, y = map(lambda z: z.to(DEVICE), (x, y))
        x.requires_grad_(True)
        y.requires_grad_(True)

        img = model(x, mask_id=mask_id, alpha=config["alpha"])
        yh = img[:, mask_id]

        loss = reduce((yh - y) ** 2, "b ... -> b", "mean")
        loss = loss.mean() / NUM_DEVICE
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    running_loss /= len(train_loader)
    scheduler.step(running_loss)
    training_display.set_description(
        "Epoch: {0}    Loss: {1:.6f}".format(i, running_loss)
    )

    """Qualitative Evaluation of Reconstruction"""
    with torch.no_grad():
        x, img, mask_id = map(lambda z: z.cpu(), (x, img, mask_id))
        x_masked = patch_fn(x)
        x_masked[:, mask_id] = 0.0
        x_masked, img = map(lambda z: z[:visual_num], (x_masked, img))
        x_masked, img = map(lambda z: patch_fn(z, reverse=True), (x_masked, img))

        torchvision.utils.save_image(
            torchvision.transforms.Resize((80, 80), antialias=False)(
                unnormalize_fn(torch.cat([x, x_masked, img], dim=0))
            ),
            IMAGE_FOLDER + "/{0}.png".format(i),
            nrow=int(visual_num**0.5),
            normalize=True,
            range=(0, 1),
        )

    torch.save(model.module.state_dict(), MODEL_FOLDER + "/model{0}.pth".format(i))
