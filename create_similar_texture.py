from pathlib import Path
import numpy as np
from WFC_train import extract_patterns, compute_pattern_occurrences, get_unique_patterns, compute_adjacencies
from WFC_generate import generate_new_level
from model import VQVAE
import argparse
from omegaconf import OmegaConf
import torchvision
import torch
from datetime import datetime
from random import sample
import uuid
import pickle


def load_texture(texture_path, conf):

    img = torchvision.io.read_image(str(texture_path), mode =torchvision.io.ImageReadMode.RGB)
    # Convert from uint8 to float32
    img = img.float()
    img = img.cuda()

    # Normalize to mean 0.5 stdv 0.5 i guess

    img = img / 255.0

    normalizer = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    resize = torchvision.transforms.Resize(conf.data.size)

    # Add a batch dimension
    img = img[None, :, :, :]

    img = resize(img)
    img = normalizer(img)
    return img

def get_texture_codes(texture_tensor, model):

    if model.conf.model.hierarchical:
        quant_t, quant_b, diff, id_t, id_b = model.encode(texture_tensor)

        return quant_t, quant_b, id_t, id_b
    else:
        quant, diff, id = model.encode(texture_tensor)

        return quant, diff, id

def train_texture_wfc(texture_codes, window_size, wrapping):

    wrapping = wrapping
    pattern_height = window_size
    pattern_width = window_size
    row_offset = 1
    col_offset = 1

    all_patterns = extract_patterns(texture_codes, pattern_height, pattern_width,
                                    row_offset=row_offset, col_offset=col_offset,
                                    wrapping=wrapping)

    pattern_occurrences = compute_pattern_occurrences(all_patterns)

    unique_patterns = get_unique_patterns(all_patterns)

    learned_adjacencies = compute_adjacencies(unique_patterns,
                                              row_offset=row_offset,
                                              col_offset=col_offset)

    trained_WFC_model = {
        "domain": "color",  # Hardcoded as color due to the way that the wfc code is set up
        "pattern_height": pattern_height,
        "pattern_width": pattern_width,
        "row_offset": row_offset,
        "col_offset": col_offset,
        "allowed_adjacencies": learned_adjacencies,
        "pattern_counts": pattern_occurrences
    }

    return trained_WFC_model

def load_model(vqvae_path):

    device = "cuda"

    model = VQVAE(conf=conf).to(device)
    model.load_state_dict(torch.load(vqvae_path))

    return model


def run_wfc_generation(trained_wfc_model, width_height, iteration_levels = 1, wrapping = False):

    level_height = width_height[1]
    level_width = width_height[0]

    new_texture_codes = generate_new_level(level_height, level_width, trained_wfc_model,
                               wrapping=wrapping, max_attempts=5, iteration_levels = iteration_levels)

    return new_texture_codes


def decode_latents(id_t, id_b, model):

    decoded = model.read_txt_and_decode_code(id_t, id_b)

    return decoded


def make_folder_structure(output_path):

    time_now = datetime.now().strftime('%m-%d-%H-%M')
    output_dir = output_path / time_now

    if not output_dir.exists():
        output_dir.mkdir()

    if not (output_dir / "wfc_sample").exists():
        (output_dir / "wfc_sample").mkdir()

    return output_dir

def __load_config():
    conf = OmegaConf.load("config.yaml")

    print(f"loaded configs: {conf}")

    return conf

def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("texture_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("output_path", type=str)

    args = parser.parse_args()

    print(args)

    return Path(args.texture_path), Path(args.model_path), Path(args.output_path)

def __save_config(conf, output_path):
    with open(output_path / "config.yaml", 'w') as f:
        OmegaConf.save(conf, f)


def float_to_heatmap_color(value, min, max):
    # https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map

    ratio = 2 * (value-min) / (max - min)
    b = torch.max(torch.zeros_like(value), 1.*(1. - ratio))
    r = torch.max(torch.zeros_like(value), 1.*(ratio - 1))
    g = 1 - b - r
    return torch.cat([r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]], 1)


def run_on_batch_of_images_from_paths(texture_paths, model, conf, output_path, num_per_texture=1):
    for texture_path in texture_paths:

        texture_tensor = load_texture(texture_path, conf)

        for idx in range(num_per_texture):

            if conf.model.hierarchical:
                quant_t, quant_b, id_t, id_b = get_texture_codes(texture_tensor, model)

                # Run WFC on texture embedding

                wfc_model_b = train_texture_wfc(texture_codes=id_b.cpu().numpy(), window_size=2, wrapping=False)
                new_id_b = run_wfc_generation(wfc_model_b, width_height=(32, 32), iteration_levels=1, wrapping=False)

                wfc_model_t = train_texture_wfc(texture_codes=id_t.cpu().numpy(), window_size=2, wrapping=False)
                new_id_t = run_wfc_generation(wfc_model_t, width_height=(16, 16), iteration_levels=1, wrapping=False)

                # Decode new latent

                # use id_t from model directly, use new_id_b for new structure latent.
                new_id_b = torch.LongTensor(new_id_b).cuda()
                new_id_t = torch.LongTensor(new_id_t).cuda()

                new_id_b = new_id_b[None, :, :]
                new_id_t = new_id_t[None, :, :]

                new_textures_b = decode_latents(id_t, new_id_b, model)
                new_textures_t = decode_latents(new_id_t, id_b, model)
                fully_new_textures = decode_latents(new_id_t, new_id_b, model)

                texture_name = texture_path.name
                img_output_path = output_path / "wfc_sample"
                img_output_path = img_output_path / f"{texture_name}.png"

                stacked_image = [texture_tensor, new_textures_b, new_textures_t, fully_new_textures]

                torchvision.utils.save_image(torch.cat(stacked_image, 0), img_output_path, normalize=True)
                torchvision.utils.save_image(torch.cat([new_textures_b, new_textures_t, fully_new_textures], 0),
                                             img_output_path, normalize=True)

            else:
                quant, diff, id = get_texture_codes(texture_tensor, model)

                # Run WFC on texture embedding

                wfc_model = train_texture_wfc(texture_codes=id.cpu().numpy(), window_size=2, wrapping=False)
                new_id = run_wfc_generation(wfc_model, width_height=(16, 16), iteration_levels=1, wrapping=False)

                # Decode new latent

                new_id = torch.LongTensor(new_id).cuda()
                new_id = new_id[None, :, :]

                new_textures = model.read_txt_and_decode_code(new_id, None)

                texture_name = texture_path.stem
                img_output_path = output_path / "wfc_sample"

                if not img_output_path.exists():
                    img_output_path.mkdir()

                timestr = datetime.now().strftime('%m-%d-%H-%M')

                original_img_output_path = img_output_path / f"{texture_name}.png"
                new_texture_img_output_path = img_output_path / f"{texture_name}_{idx}.png"

                original_code_heatmap = float_to_heatmap_color(id, 0, conf.model.codebook_size)
                original_code_heatmap = torch.nn.Upsample(size=texture_tensor.shape[2:4], mode="nearest")(original_code_heatmap.float())

                new_code_heatmap = float_to_heatmap_color(new_id, 0, conf.model.codebook_size)
                new_code_heatmap = torch.nn.Upsample(size=texture_tensor.shape[2:4], mode="nearest")(new_code_heatmap.float())

                stacked_image = [texture_tensor, original_code_heatmap, new_textures, new_code_heatmap]

                #torchvision.utils.save_image(torch.cat(stacked_image, 0), original_img_output_path, normalize=True)
                torchvision.utils.save_image(torch.cat(stacked_image, 0), new_texture_img_output_path, normalize=True)


if __name__ == "__main__":

    # read args and conf

    texture_path, model_path, output_path = __parse_args()
    output_path = make_folder_structure(output_path)
    conf = __load_config()

    # Load texture and model

    texture_paths = sample(list(texture_path.iterdir()), k=10)
    model = load_model(model_path)

    # Run WFC on texture embedding
    run_on_batch_of_images(texture_paths, model, conf, output_path, num_per_texture=3)