import torch
import numpy as np
import torchvision
from model import VQVAE
import datetime
from omegaconf import OmegaConf


# IO UTILITIES

def __load_config():
    conf = OmegaConf.load("config.yaml")

    print(f"loaded configs: {conf}")

    return conf


def make_folder_structure(output_path):

    time_now = datetime.now().strftime('%m-%d-%H-%M')
    output_dir = output_path / time_now

    if not output_dir.exists():
        output_dir.mkdir()

    if not (output_dir / "wfc_sample").exists():
        (output_dir / "wfc_sample").mkdir()

    return output_dir


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


# MODEL UTILITIES

def decode_latents(id_t, id_b, model):

    decoded = model.read_txt_and_decode_code(id_t, id_b)

    return decoded

def load_model(vqvae_path, conf):

    device = "cuda"

    model = VQVAE(conf=conf).to(device)
    model.load_state_dict(torch.load(vqvae_path))

    return model

def get_texture_codes(texture_tensor, model):

    if model.conf.model.hierarchical:
        quant_t, quant_b, diff, id_t, id_b = model.encode(texture_tensor)

        return quant_t, quant_b, id_t, id_b
    else:
        quant, diff, id = model.encode(texture_tensor)

        return quant, diff, id


 # PLOTTING UTILITIES

def float_to_heatmap_color(value, min, max):
    # https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map

    ratio = 2 * (value-min) / (max - min)
    b = torch.max(torch.zeros_like(value), 1.*(1. - ratio))
    r = torch.max(torch.zeros_like(value), 1.*(ratio - 1))
    g = 1 - b - r
    return torch.cat([r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]], 1)
