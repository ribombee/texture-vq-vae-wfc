from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from model import VQVAE
import torch
from util import get_texture_codes, float_to_heatmap_color
import torchvision

if torch.accelerator.is_available():
    device = torch.device(torch.accelerator.current_accelerator())
else:
    device = torch.device("cpu")

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

def load_vqvae(model_loc, conf):
    model = VQVAE(conf=conf).to(device)
    model.load_state_dict(torch.load(model_loc))
    model.eval()
    return model

def text_to_modeloutput(text, model):
    """
    This function takes a text and returns a model output.
    """

    texture = model.read_txt_and_decode_code(text.to(device, None))
    return texture


def __parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Read the codes from the folders in code_path and run the VQ-VAE decoder on them.")
    parser.add_argument("code_path", type=str, help="Path to the directory containing the codes")
    parser.add_argument("vqvae_path", type=str, help="Path to the directory containing the vqvae checkpoint")
    return parser.parse_args()

if __name__ == "__main__":

    args = __parse_args()
    code_path = Path(args.code_path)
    vqvae_path = Path(args.vqvae_path)

    subfolders = code_path.iterdir()
    conf = OmegaConf.load("config.yaml")

    vqvae = load_vqvae(vqvae_path, conf)

    for subfolder in subfolders:

        if not subfolder.is_dir():
            continue

        og_code = read_file_as_tensor(subfolder.parent / f"{subfolder.name}.txt")
        og_code = og_code[None, :, :]
        og_code_heatmap = float_to_heatmap_color(og_code, 0, conf.model.codebook_size)
        og_code_heatmap = torch.nn.Upsample(size=[128, 128], mode="nearest")(og_code_heatmap.float())
        og_modeloutput = text_to_modeloutput(og_code, vqvae)
        torchvision.utils.save_image(torch.cat([og_modeloutput, og_code_heatmap.to("cuda")], 0), subfolder / "original_output_from_finalthing.png", normalize=True)
        torchvision.utils.save_image(og_modeloutput, subfolder / "original_output_only_fromfinalthing.png", normalize=True)

        code_files = subfolder.rglob("*.txt.lvl")
        for code_file in code_files:
            #should only be one but idc
            code = read_file_as_tensor(code_file)
            code = code[None, :, :] # Add batch dimension
            model_output = text_to_modeloutput(code, vqvae)

            new_code_heatmap = float_to_heatmap_color(code, 0, conf.model.codebook_size)
            new_code_heatmap = torch.nn.Upsample(size=[128, 128], mode="nearest")(new_code_heatmap.float())

            torchvision.utils.save_image(torch.cat([model_output, new_code_heatmap.to(device)], 0), subfolder / f"{code_file.stem}_output.png", normalize=True)
            torchvision.utils.save_image(model_output, subfolder / f"{code_file.stem}_output_only.png", normalize=True)