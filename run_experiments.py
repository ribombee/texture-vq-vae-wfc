import argparse
from idlelib.window import register_callback

import torch
from omegaconf import OmegaConf
import torchvision.datasets
from torch.distributions import register_kl
from torchvision import transforms, datasets, utils
import pandas as pd
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchmetrics import StructuralSimilarityIndexMeasure
from util import get_texture_codes, float_to_heatmap_color
from pathlib import Path
import random
import string
from tqdm import tqdm
from model import VQVAE
from datetime import datetime
import lpips
from text_to_modeloutput import read_file_as_tensor, text_to_modeloutput

def get_test_data(data_loc):

    test_transform = transforms.Compose(
        [
            transforms.Resize(conf.data.size),
            transforms.CenterCrop(conf.data.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if not (data_loc / "test").exists():
        (data_loc / "test").mkdir()

    test_data = torchvision.datasets.DTD(data_loc / "test", split="test", download=True, transform=test_transform)

    return test_data


def export_ids_as_text(datapoint_ids, vqvae_model, out_path):
    # Add an offset to avoid control characters
    offset = 33  # Start from '!' (ASCII 33) to avoid control characters

    datapoint_ids = datapoint_ids.cpu().numpy()
    datapoint_ids = datapoint_ids[0, :, :]
    datapoint_ids = datapoint_ids.astype(int)

    with out_path.open('w', encoding='utf-8') as f:
        for row in datapoint_ids:
            for char in row:
                f.write(chr(char + offset))  # Add offset to avoid control characters
            f.write("\n")

def get_ssim():

    ssim = StructuralSimilarityIndexMeasure()
    return ssim


def get_lpips_metric():
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    def get_lpips(input, output):

        alex = loss_fn_alex(input, output)
        vgg = None # loss_fn_vgg(input, output)

        return alex, vgg

    return get_lpips

def test_model_reconstruct(idx_path, model):

    idx = read_file_as_tensor(idx_path)
    idx = idx[None, :, :] # Add batch dimension
    model_output = text_to_modeloutput(idx, model)

    torchvision.utils.save_image(model_output, idx_path.parent / idx_path.stem /f"test_model_output.png", normalize=True)

def run_experiments(reg_model, no_es_model, no_gated_model, no_ploss_model, data_loc, output_loc, conf):
    reg_model.eval()
    no_es_model.eval()
    no_gated_model.eval()
    no_ploss_model.eval()
    # Send models to cuda
    reg_model = reg_model.to("cuda")
    no_es_model = no_es_model.to("cuda")
    no_gated_model = no_gated_model.to("cuda")
    no_ploss_model = no_ploss_model.to("cuda")
    test_data = get_test_data(data_loc)
    row_list = []
    # lpips = get_lpips_metric()
    ssim = get_ssim()

    reg_output_loc = output_loc / "regular"
    no_es_output_loc = output_loc / "no_es"
    no_gated_output_loc = output_loc / "no_gated"
    no_ploss_output_loc = output_loc / "no_ploss"
    reg_output_loc.mkdir()
    no_es_output_loc.mkdir()
    no_gated_output_loc.mkdir()
    no_ploss_output_loc.mkdir()

    for test_datapoint in tqdm(test_data):
        torch.cuda.empty_cache()
        datapoint_dict = {
            "filename": None,
            "reg_psnr": None,
            "reg_ssim": None,
            "abl_no_es_psnr": None,
            "abl_no_es_ssim": None,
            "abl_no_gating_psnr": None,
            "abl_no_gating_ssim": None,
            "abl_no_ploss_psnr": None,
        }
        test_datapoint = test_datapoint[0]
        datapoint_dict["filename"] = ''.join(random.choice(string.ascii_lowercase) for i in range(16))

        output_folder = output_loc / f"{datapoint_dict['filename']}"
        reg_output_folder = reg_output_loc / f"{datapoint_dict['filename']}"
        no_es_output_folder = no_es_output_loc / f"{datapoint_dict['filename']}"
        no_gated_output_folder = no_gated_output_loc / f"{datapoint_dict['filename']}"
        no_ploss_output_folder = no_ploss_output_loc / f"{datapoint_dict['filename']}"

        if not output_folder.exists():
            output_folder.mkdir()
        if not reg_output_folder.exists():
            reg_output_folder.mkdir()
        if not no_es_output_folder.exists():
            no_es_output_folder.mkdir()
        if not no_gated_output_folder.exists():
            no_gated_output_folder.mkdir()
        if not no_ploss_output_folder.exists():
            no_ploss_output_folder.mkdir()

        with torch.no_grad():
            test_datapoint = test_datapoint.to("cuda")
            test_datapoint = test_datapoint[None, :, :, :]
            reg_model_out, reg_diff, reg_model_ids = reg_model(test_datapoint)
            no_es_model_out, no_es_diff, no_es_model_ids = no_es_model(test_datapoint)
            no_gated_model_out, no_gated_diff, no_gated_model_ids = no_gated_model(test_datapoint)
            no_ploss_model_out, no_ploss_diff, no_ploss_model_ids = abl_no_ploss_model(test_datapoint)

        reg_point_psnr = peak_signal_noise_ratio(reg_model_out, test_datapoint, 2.0)
        reg_point_ssim = ssim(test_datapoint.to("cpu"), reg_model_out.to("cpu"))
        no_es_point_psnr = peak_signal_noise_ratio(no_es_model_out, test_datapoint, 2.0)
        no_es_point_ssim = ssim(test_datapoint.to("cpu"), no_es_model_out.to("cpu"))
        no_gated_point_psnr = peak_signal_noise_ratio(no_gated_model_out, test_datapoint, 2.0)
        no_gated_point_ssim = ssim(test_datapoint.to("cpu"), no_gated_model_out.to("cpu"))
        no_ploss_point_psnr = peak_signal_noise_ratio(no_ploss_model_out, test_datapoint, 2.0)
        no_ploss_point_ssim = ssim(test_datapoint.to("cpu"), no_ploss_model_out.to("cpu"))

        datapoint_dict["reg_psnr"] = reg_point_psnr.item()
        datapoint_dict["reg_ssim"] = reg_point_ssim.item()
        datapoint_dict["abl_no_es_psnr"] = no_es_point_psnr.item()
        datapoint_dict["abl_no_es_ssim"] = no_es_point_ssim.item()
        datapoint_dict["abl_no_gating_psnr"] = no_gated_point_psnr.item()
        datapoint_dict["abl_no_gating_ssim"] = no_gated_point_ssim.item()
        datapoint_dict["abl_no_ploss_psnr"] = no_ploss_point_psnr.item()
        datapoint_dict["abl_no_ploss_ssim"] = no_ploss_point_ssim.item()

        utils.save_image(test_datapoint, output_folder / "original.png", normalize=True)
        utils.save_image(reg_model_out, output_folder / "reg_model_output.png", normalize=True)
        utils.save_image(no_es_model_out, output_folder / "no_es_model_output.png", normalize=True)
        utils.save_image(no_gated_model_out, output_folder / "no_gated_model_output.png", normalize=True)
        utils.save_image(no_ploss_model_out, output_folder / "no_ploss_model_output.png", normalize=True)

        reg_heatmap = float_to_heatmap_color(reg_model_ids, 0, conf.model.codebook_size)
        reg_heatmap_broadcast = torch.nn.Upsample(size=test_datapoint.shape[2:4], mode="nearest")(reg_heatmap.float())
        utils.save_image(reg_heatmap_broadcast, output_folder / "reg_heatmap.png", normalize=True)

        no_es_heatmap = float_to_heatmap_color(no_es_model_ids, 0, conf.model.codebook_size)
        no_es_heatmap_broadcast = torch.nn.Upsample(size=test_datapoint.shape[2:4], mode="nearest")(no_es_heatmap.float())
        utils.save_image(no_es_heatmap_broadcast, output_folder / "no_es_heatmap.png", normalize=True)

        no_gated_heatmap = float_to_heatmap_color(no_gated_model_ids, 0, conf.model.codebook_size)
        no_gated_heatmap_broadcast = torch.nn.Upsample(size=test_datapoint.shape[2:4], mode="nearest")(no_gated_heatmap.float())
        utils.save_image(no_gated_heatmap_broadcast, output_folder / "no_gated_heatmap.png", normalize=True)

        no_ploss_heatmap = float_to_heatmap_color(no_ploss_model_ids, 0, conf.model.codebook_size)
        no_ploss_heatmap_broadcast = torch.nn.Upsample(size=test_datapoint.shape[2:4], mode="nearest")(no_ploss_heatmap.float())
        utils.save_image(no_ploss_heatmap_broadcast, output_folder / "no_ploss_heatmap.png", normalize=True)

        comparison_image = torch.cat([test_datapoint, reg_model_out, no_es_model_out, no_gated_model_out, no_ploss_model_out], 0)
        utils.save_image(comparison_image, output_folder / "comparison.png", normalize=True)

        combined_image = torch.cat([reg_model_out, no_es_model_out, no_gated_model_out, reg_heatmap_broadcast, no_es_heatmap_broadcast, no_gated_heatmap_broadcast, no_ploss_heatmap_broadcast], 0)
        utils.save_image(combined_image, output_folder / "combined.png", normalize=True)

        row_list.append(datapoint_dict)
        export_ids_as_text(reg_model_ids, reg_model, reg_output_loc / f"{datapoint_dict['filename']}.txt")
        export_ids_as_text(no_es_model_ids, no_es_model,  no_es_output_loc / f"{datapoint_dict['filename']}.txt")
        export_ids_as_text(no_gated_model_ids, no_gated_model, no_gated_output_loc / f"{datapoint_dict['filename']}.txt")

    results_df = pd.DataFrame(row_list)
    results_df.to_csv(output_loc / "results.csv")


def load_model(model_loc, conf, gated=True):
    device = "cuda"
    model = VQVAE(conf=conf, gated=gated).to(device)
    model.load_state_dict(torch.load(model_loc), strict=False)
    return model


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("reg_model_path", type=str)
    parser.add_argument("abl_no_es_model_path", type=str)
    parser.add_argument("abl_no_gating_model_path", type=str)
    parser.add_argument("abl_no_ploss_model_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)

    args = parser.parse_args()

    return args.reg_model_path, args.abl_no_es_model_path, args.abl_no_gating_model_path, args.abl_no_ploss_model_path ,args.data_path, args.output_path

if __name__ == "__main__":
    reg_model_loc, no_es_model_loc, no_gating_model_loc , no_ploss_model_loc, data_loc, output_loc = __parse_args()
    data_loc = Path(data_loc)
    output_loc = Path(output_loc)
    output_loc = output_loc / datetime.now().strftime('%m-%d-%H-%M')

    if not output_loc.exists():
        output_loc.mkdir()

    conf = OmegaConf.load("config.yaml")
    reg_model = load_model(reg_model_loc, conf)
    abl_no_es_model = load_model(no_es_model_loc, conf)
    abl_no_gating_model = load_model(no_gating_model_loc, conf, gated=False)
    abl_no_ploss_model = load_model(no_ploss_model_loc, conf)

    run_experiments(reg_model, abl_no_es_model, abl_no_gating_model, abl_no_ploss_model, data_loc, output_loc, conf)