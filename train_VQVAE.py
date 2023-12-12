import argparse
import sys
import os

import torch
import torchvision.datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from omegaconf import OmegaConf
from datetime import datetime
from torchinfo import summary

from tqdm import tqdm

from model import VQVAE
# from scheduler import CycleScheduler
from math import cos, pi, floor, sin
from pathlib import Path
from torchvision.transforms import ToTensor


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cos(start, end, proportion):
    cos_val = cos(pi * proportion) + 1

    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = 0

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter
class CycleScheduler:
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        momentum=(0.95, 0.85),
        divider=25,
        warmup_proportion=0.3,
        phase=('linear', 'cos'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {'linear': anneal_linear, 'cos': anneal_cos}

        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, phase_map[phase[1]]),
        ]

        self.momentum = momentum

        if momentum is not None:
            mom1, mom2 = momentum
            self.momentum_phase = [
                Phase(mom1, mom2, phase1, phase_map[phase[0]]),
                Phase(mom2, mom1, phase2, phase_map[phase[1]]),
            ]

        else:
            self.momentum_phase = []

        self.phase = 0

    def step(self):
        lr = self.lr_phase[self.phase].step()

        if self.momentum is not None:
            momentum = self.momentum_phase[self.phase].step()

        else:
            momentum = None

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                if 'betas' in group:
                    group['betas'] = (momentum, group['betas'][1])

                else:
                    group['momentum'] = momentum

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            for phase in self.momentum_phase:
                phase.reset()

            self.phase = 0

        return lr, momentum


def train(epoch, training_loader, validation_loader, model, optimizer, scheduler, device, output_path):

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0

    model.train() # In case we have called model.eval() elsewhere.
    criterion = nn.MSELoss()
    with tqdm(training_loader, unit="batch") as tloader:
        for i, (img, _) in enumerate(tloader):
            model.zero_grad()

            img = img.to(device)

            out, latent_loss = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss
            loss.backward()

            if scheduler is not None:
                scheduler.step()
            optimizer.step()

            part_mse_sum = recon_loss.item() * img.shape[0]
            part_mse_n = img.shape[0]

            mse_sum += part_mse_sum
            mse_n += part_mse_n

            lr = optimizer.param_groups[0]["lr"]

            tloader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )


def validate(epoch, validation_loader, model, device, output_path):
    model.eval()
    with torch.no_grad():
        val_mse_sum = 0
        val_mse_n = 0
        criterion = nn.MSELoss()

        with tqdm(validation_loader, unit="batch") as vloader:
            for i, (img, _) in enumerate(vloader):
                model.zero_grad()

                img = img.to(device)

                out, latent_loss = model(img)
                recon_loss = criterion(out, img)
                latent_loss = latent_loss.mean()

                part_mse_sum = recon_loss.item() * img.shape[0]
                part_mse_n = img.shape[0]

                val_mse_sum += part_mse_sum
                val_mse_n += part_mse_n

                vloader.set_description(
                    (
                        f"epoch: {epoch + 1}; val mse: {recon_loss.item():.5f}; "
                        f"val latent: {latent_loss.item():.3f}; avg val mse: {val_mse_sum / val_mse_n:.5f}; "
                    )
                )

def plot_output(sample_tensor, model, output_path, prefix="none", epoch=-1, sample_size=25):

    with torch.no_grad():
        out, _ = model(sample_tensor)

    utils.save_image(
        torch.cat([sample_tensor, out], 0),
        output_path / f"sample/{prefix}_{str(epoch + 1).zfill(5)}.png",
        nrow=sample_size,
        normalize=True
    )

def plot_mixed_up_latents(sample_tensor, model, output_path, sample_size=25, epoch=-1):

    quant_t, quant_b, diff, id_t, id_b = model.encode(sample_tensor)

    # Take the first example image's quant_t and use every single quant_b

    single_quant_t = quant_t.clone()
    single_quant_t[:] = single_quant_t[0] # Does the broadcasting work?

    single_t_decoded = model.decode(single_quant_t, quant_b)

    # Take the first example image's quant_b and use every single quant_t

    single_quant_b = quant_b.clone()
    single_quant_b[:] = single_quant_b[0]

    single_b_decoded = model.decode(quant_t, single_quant_b)

    out, _ = model(sample_tensor)

    utils.save_image(
        torch.cat([sample_tensor, out, single_t_decoded, single_b_decoded], 0),
        output_path / f"sample/quantizer_tomfoolery_{str(epoch + 1).zfill(5)}.png",
        nrow=sample_size,
        normalize=True
    )


def plot_latent_heatmap(sample_tensor, model, output_path, sample_size=24, epoch=-1):

    quant_t, quant_b, diff, id_t, id_b = model.encode(sample_tensor)

    # TODO: implement lol
    # broadcast id_t and id_b to be the same size as the input.
    # sample_tensor is of shape (batch_size, channels, width, height)
    # id_t and id_b are of shape (batch_size, enc_width, enc_height)
    # enc_size should be a factor of size

    out, _ = model(sample_tensor)

    utils.save_image(
        torch.cat([sample_tensor, out, single_t_decoded, single_b_decoded], 0),
        output_path / f"sample/quantizer_tomfoolery_{str(epoch + 1).zfill(5)}.png",
        nrow=sample_size,
        normalize=True
    )


def make_folder_structure(output_path):

    time_now = datetime.now().strftime('%m-%d-%H-%M')
    output_dir = output_path / time_now

    if not output_dir.exists():
        output_dir.mkdir()

    if not (output_dir / "checkpoint").exists():
        (output_dir / "checkpoint").mkdir()

    if not (output_dir / "sample").exists():
        (output_dir / "sample").mkdir()

    with (output_dir / "conf.yaml").open("w") as fp:
        OmegaConf.save(conf, fp)

    return output_dir


def get_data_loaders(data_path, doom_path="data/Doom_textures"):

    transform = transforms.Compose(
        [
            transforms.RandAugment(num_ops=3, magnitude=3),
            transforms.Resize(conf.data.size),
            transforms.CenterCrop(conf.data.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(conf.data.size),
            transforms.CenterCrop(conf.data.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_path = data_path / "train"
    val_path = data_path / "validate"
    if not train_path.exists():
        train_path.mkdir()
    if not val_path.exists():
        val_path.mkdir()

    training_data = torchvision.datasets.DTD(root=str(train_path), split="train", download=True, transform=transform)
    train_loader = DataLoader(
        training_data, batch_size=conf.training.batch_size, num_workers=2, shuffle=True
    )

    val_data = torchvision.datasets.DTD(root=str(val_path), split="val", download=True, transform=val_transform)
    val_loader = DataLoader(
        val_data, batch_size=conf.training.batch_size, num_workers=2, shuffle=True
    )

    doom_dataset = datasets.ImageFolder(doom_path, transform=val_transform)
    doom_loader = DataLoader(
        doom_dataset, batch_size=conf.training.batch_size, num_workers=2
    )

    return train_loader, val_loader, doom_loader


def train_vqvae(conf, data_path, output_path):
    device = "cuda"
    output_dir = make_folder_structure(output_path) # Makes the folder structure, including a timstamped run folder
    train_loader, val_loader, doom_loader = get_data_loaders(data_path)
    model = VQVAE(conf=conf).to(device)
    print(summary(model, (conf.training.batch_size, 3, conf.data.size, conf.data.size)))
    optimizer = optim.Adam(model.parameters(), lr=conf.training.lr)
    scheduler = None
    if conf.training.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            conf.training.lr,
            n_iter=len(train_loader) * conf.training.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    train_sample = next(iter(train_loader))[0][:25].cuda()
    val_sample = next(iter(val_loader))[0][:25].cuda()
    doom_sample = next(iter(doom_loader))[0][:25].cuda()

    for i in range(conf.training.epoch):
        train(i, train_loader, val_loader, model, optimizer, scheduler, device, output_dir)
        plot_output(train_sample, model, output_dir, prefix="train", epoch=i, sample_size=25)
        validate(i, val_loader, model, device, output_dir)
        plot_output(val_sample, model, output_dir, prefix="val", epoch=i, sample_size=25)
        plot_output(doom_sample, model, output_dir, prefix="doom", epoch=i, sample_size=25)
        plot_mixed_up_latents(val_sample, model, output_dir, sample_size=25, epoch=i)

        torch.save(model.state_dict(), str(output_dir / f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt"))


def __load_config():
    conf = OmegaConf.load("config.yaml")

    print(f"loaded configs: {conf}")

    return conf

def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)

    args = parser.parse_args()

    print(args)

    return Path(args.data_path), Path(args.output_path)

if __name__ == "__main__":

    data_path, output_path = __parse_args()
    conf = __load_config()

    train_vqvae(conf, data_path, output_path)
