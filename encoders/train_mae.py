"""
train_mae.py  --  pretrain the MAE on EuroSAT train tiles (labels are never used).

Run from the project root:
    python -m encoders.train_mae --epochs 150 --bs 256 --out outputs/mae

Smoke test (a few minutes, also works on CPU):
    python -m encoders.train_mae --max-train 1024 --epochs 3 --warmup 1 --bs 128 --workers 0 --out outputs/mae_smoke

Resume: rerun the same command. If <out>/last.pt exists it continues from it
(use --fresh to ignore it). A checkpoint is written after every epoch.
Time limit: --time-limit-hours stops cleanly and saves, so a Kaggle session
cut-off never loses more than one epoch.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import get_settings
from encoders.datapipeline import get_dataloaders
from encoders.mae import MAE

settings = get_settings()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step: int, base_lr: float, warm_steps: int, total_steps: int) -> float:
    if step < warm_steps:
        return base_lr * (step + 1) / max(1, warm_steps)
    t = (step - warm_steps) / max(1, total_steps - warm_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def save_ckpt(path, model, opt, scaler, epoch, step, args):
    state = {
        "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "step": step, "args": vars(args),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a cut-off never leaves a half-written file


def save_recon(model, x, mean, std, path, device, mask_ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    x = x.to(device)
    masked, recon = model.reconstruct(x, mask_ratio)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    n = x.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5))
    for r, (title, imgs) in enumerate(zip(["original", "masked", "reconstruction"], [x, masked, recon])):
        imgs = (imgs * s + m).clamp(0, 1).cpu()
        for c in range(n):
            axes[r, c].imshow(imgs[c].permute(1, 2, 0).numpy())
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(model, val_dl, device, use_amp):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for x, _ in val_dl:
            x = x.to(device, non_blocking=True)
            loss, _, _ = model(x)
            total += loss.item() * x.shape[0]
            n += x.shape[0]
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--blr", type=float, default=1.5e-4, help="base lr, scaled by bs/256")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=10, help="warmup epochs")
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dec-dim", type=int, default=128)
    ap.add_argument("--dec-depth", type=int, default=2)
    ap.add_argument("--no-norm-pix", action="store_true", help="disable per-patch pixel normalisation in the loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=0, help="use a random subset (smoke test)")
    ap.add_argument("--time-limit-hours", type=float, default=8.0)
    ap.add_argument("--recon-every", type=int, default=25)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    ap.add_argument("--out", type=str, default="outputs/mae")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"torch {torch.__version__} | device {device} | amp {use_amp}")

    # ---- data (train split only; labels are ignored) ----
    set_seed(args.seed)
    train_dl, val_dl, info = get_dataloaders(source="eurosat", batch_size=args.bs,
                                             seed=args.seed, num_workers=args.workers)
    if args.max_train:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train_dl.dataset), generator=g)[: args.max_train].tolist()
        train_dl = DataLoader(Subset(train_dl.dataset, idx), batch_size=args.bs, shuffle=True,
                              drop_last=True, num_workers=args.workers)
    steps_per_epoch = len(train_dl)
    assert steps_per_epoch > 0, "no batches: lower --bs or raise --max-train"
    total_steps = args.epochs * steps_per_epoch
    warm_steps = args.warmup * steps_per_epoch

    # ---- model ----
    set_seed(args.seed)
    model = MAE(img_size=settings.T, patch=args.patch, in_ch=settings.C, dim=settings.D,
                depth=args.depth, heads=args.heads, dec_dim=args.dec_dim, dec_depth=args.dec_depth,
                mask_ratio=args.mask_ratio, norm_pix_loss=not args.no_norm_pix).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"params: total {n_params:,} | encoder {n_enc:,} | embedding dim {settings.D}")

    base_lr = args.blr * args.bs / 256
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.endswith("mask_token")]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 or n.endswith("mask_token")]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=base_lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- resume ----
    start_epoch, step = 0, 0
    last = out / "last.pt"
    if last.exists() and not args.fresh:
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        torch.set_rng_state(ck["rng_cpu"])
        if ck["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        print(f"resumed from {last}: continuing at epoch {start_epoch} (step {step})")
        if start_epoch >= args.epochs:
            print("training already finished for this --epochs; nothing to do")

    # fixed validation tiles for reconstruction pictures, spread across classes
    val_ds = val_dl.dataset
    vis_idx = torch.linspace(0, len(val_ds) - 1, 8).long().tolist()
    x_vis = torch.stack([val_ds[i][0] for i in vis_idx])

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        te = time.time()
        run, nb = 0.0, 0
        for x, _ in train_dl:
            x = x.to(device, non_blocking=True)
            lr = lr_at(step, base_lr, warm_steps, total_steps)
            for g_ in opt.param_groups:
                g_["lr"] = lr
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, _, _ = model(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            run += loss.item()
            nb += 1
        train_loss = run / nb
        val_loss = evaluate(model, val_dl, device, use_amp)
        secs = time.time() - te
        print(f"epoch {epoch + 1}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| lr {lr:.2e} | {secs:.0f}s")

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_log:
                w.writerow(["epoch", "step", "train_loss", "val_loss", "lr", "seconds"])
                new_log = False
            w.writerow([epoch + 1, step, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{lr:.3e}", f"{secs:.1f}"])

        save_ckpt(last, model, opt, scaler, epoch, step, args)

        last_epoch = epoch == args.epochs - 1
        if (epoch + 1) % args.recon_every == 0 or epoch == start_epoch or last_epoch:
            save_recon(model, x_vis, info["mean"], info["std"],
                       out / f"recon_ep{epoch + 1:03d}.png", device, args.mask_ratio)

        if not last_epoch and (time.time() - t0) > args.time_limit_hours * 3600:
            print(f"time limit of {args.time_limit_hours} h reached. Checkpoint saved at {last}. "
                  f"Rerun the same command to resume.")
            return

    # ---- final artefacts: encoder only, plus everything needed to reuse it ----
    torch.save({
        "encoder": model.encoder.state_dict(),
        "config": {"patch": args.patch, "depth": args.depth, "heads": args.heads, "dim": settings.D,
                   "img_size": settings.T, "in_ch": settings.C, "mask_ratio": args.mask_ratio,
                   "norm_pix_loss": not args.no_norm_pix, "epochs": args.epochs, "seed": args.seed},
        "norm": {"source": "eurosat", "mean": [float(v) for v in info["mean"]],
                 "std": [float(v) for v in info["std"]]},
    }, out / "encoder.pt")
    print(f"done. encoder weights saved to {out / 'encoder.pt'}")


if __name__ == "__main__":
    main()