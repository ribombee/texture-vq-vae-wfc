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
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from model import VQVAE
# from scheduler import CycleScheduler
from math import cos, pi, floor, sin
from pathlib import Path
import torchmetrics
import torchvision.transforms as transforms



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

def get_num_latent_loss(codebook_size):
    def num_latents_loss(latents):
        # latents is a list of tensors of shape (batch_size, num_latents, latent_size)
        # We want to calculate the number of unique latents in each tensor, and return the sum of all of them.
        # The purpose of this loss is to decrease the number of latents used by the encoder in order to be able to run WFC.

        flat_latents = torch.flatten(latents, start_dim=1, end_dim=2)
        #flat_latents = flat_latents.swapaxes(1, 2)

        loss = 0

        for batch_img in flat_latents:

            loss += batch_img.unique().numel() / codebook_size

        loss = loss / flat_latents.shape[0]

        # unique_tensors = flat_latents.unique(dim=1) # dim 1 should be the flattened 16*16 vectors of size 32
        return loss

    return num_latents_loss

def train(epoch, training_loader, validation_loader, model, optimizer, scheduler, device, output_path, writer, conf, hierarchical=False):

    latent_loss_weight = 1
    latent_count_loss_weight = conf.training.count_loss_weight

    mse_sum = 0
    mse_n = 0

    model.train() # In case we have called model.eval() elsewhere.
    criterion = nn.MSELoss()
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    num_latent_loss = get_num_latent_loss(model.conf.model.codebook_size)
    with tqdm(training_loader, unit="batch") as tloader:
        for i, (img, _) in enumerate(tloader):
            model.zero_grad()

            img = img.to(device)
            if hierarchical:
                out, latent_loss, id_t, id_b = model(img)
            else:
                out, latent_loss, id = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            if hierarchical:
                latent_count_loss_b = num_latent_loss(id_t)
                latent_count_loss_t = num_latent_loss(id_b)
                latent_count_loss = latent_count_loss_b + latent_count_loss_t
            else:
                latent_count_loss = num_latent_loss(id)
            # latent_count_loss = latent_count_loss.mean()

            loss = recon_loss + latent_loss_weight * latent_loss + latent_count_loss * latent_count_loss_weight
            ssim_value = ssim_metric(out, img)

            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Recon_Loss/train", recon_loss, epoch)
            writer.add_scalar("Latent_Loss/train", latent_loss, epoch)
            writer.add_scalar("Latent_Count_Loss/train", latent_count_loss, epoch)
            writer.add_scalar("SSIM/train", ssim_value.item(), epoch)

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
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; ssim: {ssim_value.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"latent_count: {latent_count_loss:.3f};"
                    f"lr: {lr:.9f}"
                )
            )


def validate(epoch, validation_loader, model, device, output_path, writer, hierarchical=False):
    model.eval()
    conf = model.conf
    with torch.no_grad():
        val_mse_sum = 0
        val_mse_n = 0
        val_latent_sum = 0
        val_latent_count_sum = 0
        criterion = nn.MSELoss()
        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        num_latent_loss = get_num_latent_loss(model.conf.model.codebook_size)

        with tqdm(validation_loader, unit="batch") as vloader:
            for i, (img, _) in enumerate(vloader):
                model.zero_grad()

                img = img.to(device)

                if hierarchical:
                    out, latent_loss, id_t, id_b = model(img)
                else:
                    out, latent_loss, id = model(img)
                recon_loss = criterion(out, img)
                latent_loss = latent_loss.mean()

                part_mse_sum = recon_loss.item() * img.shape[0]
                part_mse_n = img.shape[0]
                part_latent_sum = latent_loss.item() * img.shape[0]

                val_mse_sum += part_mse_sum
                val_mse_n += part_mse_n
                val_latent_sum += part_latent_sum

                ssim_value = ssim_metric(out, img)

                if hierarchical:
                    latent_count_loss_b = num_latent_loss(id_t)
                    latent_count_loss_t = num_latent_loss(id_b)
                    latent_count_loss = latent_count_loss_b + latent_count_loss_t
                else:
                    latent_count_loss = num_latent_loss(id)

                part_latent_count_sum = latent_count_loss * img.shape[0]
                val_latent_count_sum += part_latent_count_sum

                out_avg = out.mean()
                in_avg = img.mean()
                out_max = out.max()
                in_max = img.max()

                writer.add_scalar("Recon_Loss/val", recon_loss, epoch)
                writer.add_scalar("Latent_Loss/val", latent_loss, epoch)
                writer.add_scalar("Latent_Count_Loss/val", latent_count_loss, epoch)
                writer.add_scalar("SSIM/val", ssim_value.item(), epoch)

                vloader.set_description(
                    (
                        f"epoch: {epoch + 1}; val mse: {recon_loss.item():.5f}; val ssim: {ssim_value.item():.5f}; "
                        f"val latent: {latent_loss.item():.3f}; avg val mse: {val_mse_sum / val_mse_n:.5f}; "
                        f"val latent_count: {latent_count_loss:.3f}"
                        f"val out_avg: {out_avg:.3f}; val in_avg: {in_avg:.3f}; val out_max: {out_max:.3f}; val in_max: {in_max:.3f};"
                        f"val stop criteria: {(val_mse_sum + val_latent_sum + val_latent_count_sum) / val_mse_n:.5f}"                    )
                )
        return (val_mse_sum) / val_mse_n

def plot_output(sample_tensor, model, output_path, prefix="none", epoch=-1, sample_size=25):

    with torch.no_grad():
        if model.conf.model.hierarchical:
            out, _, _, _ = model(sample_tensor)
        else:
            out, _, _ = model(sample_tensor)

    utils.save_image(
        torch.cat([sample_tensor, out], 0),
        output_path / f"sample/{prefix}_{str(epoch + 1).zfill(5)}.png",
        nrow=sample_size,
        normalize=True
    )

def plot_mixed_up_latents(sample_tensor, model, output_path, sample_size=25, epoch=-1):
    # NOTE: this is for a hierarchical model.

    quant_t, quant_b, diff, id_t, id_b = model.encode(sample_tensor)

    # Take the first example image's quant_t and use every single quant_b

    single_quant_t = quant_t.clone()
    single_quant_t[:] = single_quant_t[0] # Does the broadcasting work?

    single_t_decoded = model.decode(single_quant_t, quant_b)

    # Take the first example image's quant_b and use every single quant_t

    single_quant_b = quant_b.clone()
    single_quant_b[:] = single_quant_b[0]

    single_b_decoded = model.decode(quant_t, single_quant_b)

    out, _, _, _ = model(sample_tensor)

    utils.save_image(
        torch.cat([sample_tensor, out, single_t_decoded, single_b_decoded], 0),
        output_path / f"sample/quantizer_tomfoolery_{str(epoch + 1).zfill(5)}.png",
        nrow=sample_size,
        normalize=True
    )


def float_to_heatmap_color(value, min, max):
    # https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map

    ratio = 2 * (value-min) / (max - min)
    b = torch.max(torch.zeros_like(value), 1.*(1. - ratio))
    r = torch.max(torch.zeros_like(value), 1.*(ratio - 1))
    g = 1 - b - r
    return torch.cat([r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]], 1)

def plot_latent_heatmap(sample_tensor, model, output_path, codebook_size, sample_size=24, epoch=-1, hierarchical=False):

    if hierarchical:
        quant_t, quant_b, diff, id_t, id_b = model.encode(sample_tensor)

        out, _, _, _ = model(sample_tensor)

        original_tensor_shape = sample_tensor.shape[2:4]

        id_t = float_to_heatmap_color(id_t, 0, codebook_size)
        id_b = float_to_heatmap_color(id_b, 0, codebook_size)

        id_t_broadcast = torch.nn.Upsample(size=original_tensor_shape, mode="nearest")(id_t.float())
        id_b_broadcast = torch.nn.Upsample(size=original_tensor_shape, mode="nearest")(id_b.float())

        utils.save_image(
            torch.cat([sample_tensor, out, id_t_broadcast, id_b_broadcast], 0),
            output_path / f"sample/quantizer_tomfoolery_{str(epoch + 1).zfill(5)}.png",
            nrow=sample_size,
            normalize=True
        )
    else:
        quant, diff, id = model.encode(sample_tensor)

        out, _, _ = model(sample_tensor)

        original_tensor_shape = sample_tensor.shape[2:4]

        id = float_to_heatmap_color(id, 0, codebook_size)

        id_broadcast = torch.nn.Upsample(size=original_tensor_shape, mode="nearest")(id.float())

        utils.save_image(
            torch.cat([sample_tensor, out, id_broadcast], 0),
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

def get_dtd_data_loaders(data_path, doom_path="data/Doom_textures"):
    transform = transforms.Compose(
        [
            transforms.RandAugment(),
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

def get_early_stopper(patience, min_delta):

    lowest_loss = float("inf")
    def early_stopper(val_losses):
        if len(val_losses) < patience:
            return False

        if all(val_losses[-patience] - val_losses[-i] < min_delta for i in range(1, patience)):
            return True

        return False

    return early_stopper


def train_vqvae(conf, data_path, output_path):
    device = "cuda"
    output_dir = make_folder_structure(output_path)
    train_loader, val_loader, doom_loader = get_dtd_data_loaders(data_path)
    model = VQVAE(conf=conf, gated=False).to(device)
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

    encoder_early_stopper = get_early_stopper(conf.training.es_patience, conf.training.es_min_delta)
    encoder_stopped = False
    encoder_stopped_at = 0
    val_losses = []

    train_sample = next(iter(train_loader))[0][:25].cuda()
    val_sample = next(iter(val_loader))[0][:25].cuda()
    doom_sample = next(iter(doom_loader))[0][:25].cuda()

    if not (output_dir / "logs").exists():
        (output_dir / "logs").mkdir()

    for i in range(conf.training.epoch):
        writer = SummaryWriter(log_dir=output_dir / "logs")
        train(i, train_loader, val_loader, model, optimizer, scheduler, device, output_dir, writer, conf)
        plot_output(train_sample, model, output_dir, prefix="train", epoch=i, sample_size=25)
        val_loss = validate(i, val_loader, model, device, output_dir, writer)
        val_losses.append(val_loss)
        plot_output(val_sample, model, output_dir, prefix="val", epoch=i, sample_size=25)
        plot_output(doom_sample, model, output_dir, prefix="doom", epoch=i, sample_size=25)
        plot_latent_heatmap(val_sample, model, output_dir, codebook_size=conf.model.codebook_size, sample_size=25, epoch=i)

        writer.flush()

        torch.save(model.state_dict(), str(output_dir / f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt"))

        if i > conf.training.es_begin:
            if not encoder_stopped:
                if encoder_early_stopper(val_losses):
                    for param in model.enc_b.parameters():
                        param.requires_grad = False
                    encoder_stopped = True
                    encoder_stopped_at = i


    print(f"Encoder stopped at epoch {encoder_stopped_at}")

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
