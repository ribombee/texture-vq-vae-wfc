
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from torchvision import utils
from model import VQVAE
from image_diversity import ClipMetrics
from omegaconf import OmegaConf
import time
import torchvision.models as models

from nca import *


def load_model(model_loc, conf, gated=True):
    device = "cuda"
    model = VQVAE(conf=conf, gated=gated).to(device)
    model.load_state_dict(torch.load(model_loc))
    return model

def perturb_latent(quant, stdev: float):
    """
    batch_size, height, width = quant.shape
    perturbed_quant = quant.clone()

    for i in range(batch_size):
        for j in range(height):
            for k in range(width):
                perturbed_quant[i, j, k] = perturbed_quant[i, j, k] * (1 + (torch.rand(1) * 2 - 1) * stdev)
    """
    noise = torch.normal(mean=0.0, std=stdev, size=quant.shape).to(quant.device)
    perturbed_quant = quant + noise

    return perturbed_quant

def read_file_as_tensor(file_path, line_length=16):
    """
    This function reads a file and returns its content as a pytorch tensor.
    """

    parsed_content = []

    with open(file_path, "r") as file:
        content = file.read()

        for row_idx in range(line_length):
            row_content = []
            for char_idx in range(line_length):
                # row idx is added to get rid of the newline characters
                idx = ord(content[row_idx * line_length + row_idx + char_idx])
                row_content.append(idx)
            parsed_content.append(row_content)

    return torch.LongTensor(parsed_content)

def read_txt_perturb_and_decode(code_path, model, std_dev=0.5):
    code = read_file_as_tensor(code_path, 16)
    code = code[None, :, :]
    code = torch.LongTensor(code).cuda()
    quant = model.quantize_t.embed_code(code)
    quant = quant.permute(0, 3, 1, 2)
    perturbed_code = perturb_latent(quant, std_dev)
    perturbed_img = model.decode(perturbed_code, None)
    return perturbed_img

def read_txt_and_decode_code(code_path, model):
    code = read_file_as_tensor(code_path, 16)
    code = code[None, :, :]
    code = torch.LongTensor(code).cuda()

    decoded_img = model.decode_code(code, None)
    return decoded_img

def run_diversity_experiments(vqvae_model, no_es_model, no_gated_model, data_loc, output_loc, conf):


    vqvae_model.eval()
    no_es_model.eval()
    no_gated_model.eval()
    vgg16 = models.vgg16(weights='IMAGENET1K_V1').features.to("cuda")
    # Not needed for now, itll probably be better to run this in a 4th eval script
    # resnet_50 = models.resnet50(weights='IMAGENET1K_V1').to("cuda")
    # resnet_50.eval() # Used for the classifier evaluation bit.

    data_loc = Path(data_loc)
    output_loc = Path(output_loc)
    nca_image_loc = output_loc / "nca_images"
    abl_no_es_loc = output_loc / "no_es"
    abl_no_gated_loc = output_loc / "no_gating"
    abl_no_es_image_loc = output_loc / "no_es_images"
    abl_no_gated_image_loc = output_loc / "no_gating_images"

    if not nca_image_loc.exists():
        nca_image_loc.mkdir()
    perturbed_image_loc = output_loc / "perturbed_images"
    if not perturbed_image_loc.exists():
        perturbed_image_loc.mkdir()
    our_image_loc = output_loc / "our_images"
    if not our_image_loc.exists():
        our_image_loc.mkdir()
    if not abl_no_es_image_loc.exists():
        abl_no_es_image_loc.mkdir()
    if not abl_no_gated_image_loc.exists():
        abl_no_gated_image_loc.mkdir()

    all_code_paths = list(data_loc.rglob("*.txt"))
    row_list = []
    clip_metrics = ClipMetrics(n_eigs=3)

    for code_path in tqdm(all_code_paths):
        print(f"Processing {code_path}")
        img_name = code_path.stem
        output_folder = output_loc / img_name
        if not output_folder.exists():
            output_folder.mkdir()

        original_img = read_txt_and_decode_code(code_path, vqvae_model)
        utils.save_image(original_img, output_folder / "original.png", normalize=True)

        original_model_generated_images = []
        no_es_generated_images = []
        no_gated_generated_images = []
        perturbed_images = []
        normal_successes = 0
        no_es_successes = 0
        no_gated_successes = 0
        for idx in range(5):
            # Normal model

            new_code_path = (code_path.parent / code_path.stem) / f"new_{idx}.txt.lvl"
            if new_code_path.exists():
                normal_successes+= 1
                generated_img = read_txt_and_decode_code(new_code_path, vqvae_model)
                original_model_generated_images.append(generated_img)
                utils.save_image(generated_img, our_image_loc / f"generated_{idx}.png", normalize=True)

            # Ablation model with no early stopping
            new_code_path = (abl_no_es_loc / code_path.stem) / f"new_{idx}.txt.lvl"
            if new_code_path.exists():
                no_es_successes+= 1
                generated_img = read_txt_and_decode_code(new_code_path, no_es_model)
                no_es_generated_images.append(generated_img)
                utils.save_image(generated_img, abl_no_es_image_loc / f"generated_{idx}.png", normalize=True)

            # Ablation model with no gated convolutions
            new_code_path = (abl_no_gated_loc / code_path.stem) / f"new_{idx}.txt.lvl"
            if new_code_path.exists():
                no_gated_successes+= 1
                generated_img = read_txt_and_decode_code(new_code_path, no_gated_model)
                no_gated_generated_images.append(generated_img)
                utils.save_image(generated_img, abl_no_gated_image_loc / f"generated_{idx}.png", normalize=True)

            # These should not fail, so no need to check for existence
            perturbed_img = read_txt_perturb_and_decode(code_path, vqvae_model)
            perturbed_images.append(perturbed_img)
            utils.save_image(perturbed_img, perturbed_image_loc / f"perturbed_{idx}.png", normalize=True)

        print(f"Finished VQVAE generation for {code_path}")

        # Train and generate images using NCA model
        start_time = time.time()
        nca_model = train_nca_model(original_img.to("cuda"), vgg16)
        nca_train_time = time.time() - start_time
        nca_images = []
        nca_generate_times = []
        for idx in range(5):
            start_time = time.time()
            nca_img = nca_model.generate_image(original_img.to("cuda"))
            nca_gen_time = time.time() - start_time
            nca_generate_times.append(nca_gen_time)
            nca_images.append(nca_img)
            utils.save_image(nca_img, nca_image_loc / f"nca_generated_{idx}.png", normalize=True)

        print(f"Finished NCA generation for {code_path}")

        # Save comparison images
        baseline_comparison_image = torch.cat(original_model_generated_images + nca_images + perturbed_images, 0)
        utils.save_image(baseline_comparison_image, output_folder / "baseline_comparison.png", normalize=True, nrow=5)

        ablation_comparison_image = torch.cat(original_model_generated_images + no_es_generated_images + no_gated_generated_images, 0)
        utils.save_image(ablation_comparison_image, output_folder / "ablation_comparison.png", normalize=True, nrow=5)

        # Classification evaluation.

        row_list.append({
            "filename": img_name,
            "successful_normal": normal_successes,
            "successful_abl_noes": no_es_successes,
            "successful_abl_nogate": no_gated_successes,
            "nca_train_time": nca_train_time,
            "nca_generate_time": np.mean(nca_generate_times),
        })

    our_tce = clip_metrics.tce(str(our_image_loc))
    nca_tce = clip_metrics.tce(str(nca_image_loc))
    perturbed_tce = clip_metrics.tce(str(perturbed_image_loc))
    no_es_tce = clip_metrics.tce(str(abl_no_es_image_loc))
    no_gated_tce = clip_metrics.tce(str(abl_no_gated_image_loc))

    print(f"Our TCE: {our_tce}")
    print(f"NCA TCE: {nca_tce}")
    print(f"Perturbed TCE: {perturbed_tce}")
    print(f"No ES TCE: {no_es_tce}")
    print(f"No Gated TCE: {no_gated_tce}")
    results_df = pd.DataFrame(row_list)
    results_df.to_csv(output_loc / "diversity_results.csv")

def __parse_args():
    parser = argparse.ArgumentParser(description="Run diversity experiments")
    parser.add_argument("vqvae_model_path", type=str, help="Path to the VQ-VAE model")
    parser.add_argument("no_es_model_path", type=str, help="Path to the no early stopping model")
    parser.add_argument("no_gated_model_path", type=str, help="Path to the no gated convolutions model")
    parser.add_argument("data_path", type=str, help="Path to the directory containing the codes")
    parser.add_argument("output_path", type=str, help="Path to the output directory")
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_default_device("cuda")
    args = __parse_args()
    conf = OmegaConf.load("config.yaml")
    vqvae_model = load_model(args.vqvae_model_path, conf)
    no_es_model = load_model(args.no_es_model_path, conf)
    no_gated_model = load_model(args.no_gated_model_path, conf, gated=False)
    run_diversity_experiments(vqvae_model, no_es_model, no_gated_model, args.data_path, args.output_path, conf)