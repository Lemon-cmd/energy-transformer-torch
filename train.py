import os, tqdm
import torch
import argparse
import pprint
import numpy as np
from einops import reduce
from torch.utils.data.dataloader import DataLoader

from torchvision.transforms import Resize, InterpolationMode
from torchvision.utils import save_image

from image_et import (
    ImageET as ET,
    Patch,
    GetCIFAR,
    gen_mask_id,
    count_parameters,
    device,
    str2bool,
    get_latest_file,
)

from time import time
from accelerate import Accelerator


def make_dir(dir_name: str):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def main(args):
    IMAGE_FOLDER = args.result_dir + "/images"
    MODEL_FOLDER = args.result_dir + "/models"

    accelerator = Accelerator()
    device = accelerator.device

    make_dir(args.result_dir)
    make_dir(IMAGE_FOLDER)
    make_dir(MODEL_FOLDER)

    x = torch.randn(1, 3, 32, 32)
    patch_fn = Patch(dim=args.patch_size)
    model = ET(
        x,
        patch_fn,
        args.out_dim,
        args.tkn_dim,
        args.qk_dim,
        args.nheads,
        args.hn_mult,
        args.attn_beta,
        args.attn_bias,
        args.hn_bias,
        time_steps=args.time_steps,
        blocks=args.blocks,
        hn_fn=lambda x: -0.5 * (torch.relu(x) ** 2.0).sum(),
    )

    if accelerator.is_main_process:
        print(f"Number of parameters: {count_parameters(model)}", flush=True)

    NUM_PATCH = model.patch.N
    NUM_MASKS = int(model.patch.N * args.mask_ratio)
    trainset, testset, unnormalize_fn = GetCIFAR(args.data_path, args.data_name)

    train_loader, test_loader = map(
        lambda z: DataLoader(
            z,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        ),
        (trainset, testset),
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        cooldown=0,
        patience=10,
        factor=0.85,
    )

    start_epoch = 1
    latest_checkpoint = get_latest_file(MODEL_FOLDER, ".pth")
    if latest_checkpoint is not None:
        if accelerator.is_main_process:
            print(f"Found latest checkpoint file: {latest_checkpoint}", flush=True)

        checkpoint = torch.load(latest_checkpoint, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        opt.load_state_dict(checkpoint["opt"])
        model.load_state_dict(checkpoint["model"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    model, opt, train_loader, test_loader, scheduler = accelerator.prepare(
        model, opt, train_loader, test_loader, scheduler
    )

    visual_num = 16
    training_display = range(start_epoch, args.epochs + 1)

    for i in training_display:
        running_loss = 0.0

        model.train()

        start_time = time()
        for x, _ in train_loader:
            # grab mask indices
            batch_id, mask_id = gen_mask_id(NUM_PATCH, NUM_MASKS, x.size(0))

            # grab the supposed-masked tokens as labels
            y = patch_fn(x)[batch_id, mask_id]

            x, y, batch_id, mask_id = map(
                lambda z: z.to(device), (x, y, batch_id, mask_id)
            )

            pred = model(x, mask=(batch_id, mask_id), alpha=args.alpha)

            # grab recovered mask-tokens
            yh = pred[batch_id, mask_id]

            loss = reduce((yh - y) ** 2, "b ... -> b", "mean").mean()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            opt.zero_grad()
            running_loss += loss.item()

        torch.cuda.synchronize()
        end_time = time()
        running_loss /= len(train_loader)
        scheduler.step(running_loss)

        if accelerator.is_main_process:
            epoch_time = end_time - start_time
            avg_loss = torch.tensor(running_loss / len(train_loader), device=device)
            avg_loss = avg_loss / accelerator.num_processes
            print(
                f"Epoch: {i}/{args.epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.5f}s",
                flush=True,
            )

        if i % args.ckpt_every == 0:
            if accelerator.is_main_process:
                with torch.no_grad():
                    x, pred, batch_id, mask_id = map(
                        lambda z: z.cpu(), (x, pred, batch_id, mask_id)
                    )
                    x_masked = patch_fn(x)
                    x_masked[batch_id, mask_id] = 0.0
                    x, x_masked, pred = map(
                        lambda z: z[:visual_num], (x, x_masked, pred)
                    )
                    x_masked, pred = map(
                        lambda z: patch_fn(z, reverse=True), (x_masked, pred)
                    )
                    img = Resize((64, 64), antialias=True)(
                        torch.cat([x, x_masked, pred], dim=0)
                    )
                    img = unnormalize_fn(img)

                    save_image(
                        img,
                        IMAGE_FOLDER + "/{0}.png".format(i),
                        nrow=4,
                        normalize=True,
                        scale_each=True,
                    )

                ckpt = {
                    "epoch": i + 1,
                    "model": model.module.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
                torch.save(ckpt, MODEL_FOLDER + f"/{i}.pth")
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ET as Mask Auto-encoder")
    parser.add_argument("--global-seed", default=3407, type=int)
    parser.add_argument("--ckpt-every", default=1, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--qk-dim", default=64, type=int)
    parser.add_argument("--mask-ratio", default=0.85, type=float)
    parser.add_argument("--blocks", default=1, type=int)
    parser.add_argument("--out-dim", default=None, type=int)
    parser.add_argument("--tkn-dim", default=256, type=int, help="token dimension")
    parser.add_argument("--nheads", default=12, type=int)
    parser.add_argument("--attn-beta", default=None, type=float)
    parser.add_argument("--hn-mult", default=4.0, type=float)
    parser.add_argument(
        "--alpha", default=1.0, type=float, help="step size for ET's dynamic"
    )
    parser.add_argument("--attn-bias", default=False, type=str2bool)
    parser.add_argument("--hn-bias", default=False, type=str2bool)
    parser.add_argument(
        "--time-steps", default=12, type=int, help="number of timesteps for ET"
    )
    parser.add_argument("--result-dir", default="./results", type=str)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=8e-5, type=float, help="learning rate")
    parser.add_argument("--b1", default=0.9, type=float, help="adam beta1")
    parser.add_argument("--b2", default=0.999, type=float, help="adam beta2")

    parser.add_argument(
        "--avg-gpu",
        default=True,
        type=str2bool,
        help="a flag indicating to divide loss by the number of devices",
    )

    parser.add_argument(
        "--weight-decay", default=0.001, type=float, help="weight decay value"
    )

    parser.add_argument(
        "--data-path", default="./", type=str, help="root folder of dataset"
    )

    parser.add_argument(
        "--data-name", default="cifar10", type=str, choices=["cifar10", "cifar100"]
    )
    args = parser.parse_args()
    main(args)
