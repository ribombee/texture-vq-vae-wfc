
# This script expects run_sturgeon_on_codes to have been run.
# It will read the files in the directory and run the VQ-VAE decoder on them.
# It will also run the ablation models, the one with no early stopping and the one with no gated convolutions.
# It will also run the perturbed baseline version of the VQ-VAE (based on experiment_perturbed_latent.py).
# And further it will train and run the NCA baseline model.
# This is the NCA model: https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb#scrollTo=3pxjdESnYlIC
# With each of these three methods, five novel images will be created per datapoint in the test set.
# For each approach, the images will be saved in the output direectory.
# Diversity will be calculated using truncated CLIP entropy from image-diversity.
# The diversity results will then be saved in the output directory.
# For each datapoint, a classifier model will be run on each of the five generated images for each method.
# The classifier model will also be run on the original image and this will be used as the truth value for calculating accuracy and f1 score.
# The accuracy and the F1 score of the classifier model will be saved in the output directory.
# We will use this classifier: https://github.com/scabini/RADAM/blob/main/standalone_RADAM_example.py
# TODO: check that i followed my recipe
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
import torch.nn.functional as F
from torch import nn
from torchvision import models

# NCA Model Implementation
class CA(nn.Module):
    def __init__(self, chn=12, hidden_n=96):
        super().__init__()
        self.chn = chn
        self.w1 = nn.Conv2d(chn * 4, hidden_n, 1)
        self.w2 = nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()

    def forward(self, x, update_rate=0.5):
        y = self.perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w) + update_rate).floor()
        return x + y * update_mask

    def perception(self, x):
        ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])
        filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
        return self.perchannel_conv(x, filters).to("cuda")

    def perchannel_conv(self, x, filters):
        b, ch, h, w = x.shape
        y = x.reshape(b * ch, 1, h, w)
        y = F.pad(y, [1, 1, 1, 1], 'circular')
        y = F.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)

    def seed(self, n, sz=128):
        return torch.zeros(n, self.chn, sz, sz)

def to_rgb(x):
    return x[..., :3, :, :] + 0.5

def calc_styles_vgg(imgs, vgg16):
  style_layers = [1, 6, 11, 18, 25]
  mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None].to("cuda")
  std = torch.tensor([0.229, 0.224, 0.225])[:,None,None].to("cuda")
  x = (imgs-mean) / std
  b, c, h, w = x.shape
  features = [x.reshape(b, c, h*w)]
  for i, layer in enumerate(vgg16[:max(style_layers)+1]):
    x = layer(x)
    if i in style_layers:
      b, c, h, w = x.shape
      features.append(x.reshape(b, c, h*w))
  return features

def project_sort(x, proj):
  return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

def ot_loss(source, target, proj_n=32):
  ch, n = source.shape[-2:]
  projs = F.normalize(torch.randn(ch, proj_n), dim=0)
  source_proj = project_sort(source, projs)
  target_proj = project_sort(target, projs)
  target_interp = F.interpolate(target_proj, n, mode='nearest')
  return (source_proj-target_interp).square().sum()

def create_vgg_loss(target_img, vgg16):
  yy = calc_styles_vgg(target_img, vgg16)
  def loss_f(imgs):
    xx = calc_styles_vgg(imgs, vgg16)
    return sum(ot_loss(x, y) for x, y in zip(xx, yy))
  return loss_f

def project_sort(x, proj):
    return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

def ot_loss(source, target, proj_n=32):
    ch, n = source.shape[-2:]
    projs = F.normalize(torch.randn(ch, proj_n), dim=0)
    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)
    target_interp = F.interpolate(target_proj, n, mode='nearest')
    return (source_proj - target_interp).square().sum()

def generate_image(self, original_img, steps=100):
    x = self.seed(1, original_img.shape[-1]).cuda()
    for _ in range(steps):
        x = self(x)
    return to_rgb(x)

def train_nca_model(original_img, vgg16, num_steps=5000):
    ca = CA()
    opt = torch.optim.Adam(ca.parameters(), 1e-3, capturable=True)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1000, 2000], 0.3)
    loss_log = []
    with torch.no_grad():
        pool = ca.seed(256)
    loss_f = create_vgg_loss(original_img, vgg16)

    for i in range(num_steps):
        with torch.no_grad():
            batch_idx = np.random.choice(len(pool), 4, replace=False)
            x = pool[batch_idx].to("cuda")
            if i % 8 == 0:
                x[:1] = ca.seed(1)
        step_n = np.random.randint(32, 96)
        for k in range(step_n):
            x = ca(x)

        overflow_loss = (x - x.clamp(-1.0, 1.0)).abs().sum()
        loss = loss_f(to_rgb(x)) + overflow_loss
        loss.backward()
        for p in ca.parameters():
            p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        opt.zero_grad()
        lr_sched.step()
        with torch.no_grad():
            pool[batch_idx] = x
        loss_log.append(loss.item())
    return ca

def load_model(model_loc, conf, gated=True):
    device = "cuda"
    model = VQVAE(conf=conf, gated=gated).to(device)
    model.load_state_dict(torch.load(model_loc))
    return model

def perturb_latent(quant, max_percentage_perturb: float):
    """
    Multiply the latent vectors in quant by a random amount, up to 1 plus or minus the maximum defined by max_percentage_perturb.
    :param quant: a batch by height by width tensor of latent vectors
    :param max_percentage_perturb: the maximum percentage by which to perturb each latent vector
    :return:
    """
    batch_size, height, width = quant.shape
    perturbed_quant = quant.clone()

    for i in range(batch_size):
        for j in range(height):
            for k in range(width):
                perturbed_quant[i, j, k] = perturbed_quant[i, j, k] * (1 + (torch.rand(1) * 2 - 1) * max_percentage_perturb)

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

def read_txt_perturb_and_decode(code_path, model, max_percentage_perturb=0.1):
    code = read_file_as_tensor(code_path, 16)
    code = code[None, :, :]
    code = torch.LongTensor(code).cuda()
    perturbed_code = perturb_latent(code, max_percentage_perturb)
    perturbed_img = model.decode_code(perturbed_code, None)
    return perturbed_img

def read_txt_and_decode_code(code_path, model):
    code = read_file_as_tensor(code_path, 16)
    code = code[None, :, :]
    code = torch.LongTensor(code).cuda()
    reconstruction = model.decode_code(code, None)
    return reconstruction

def run_diversity_experiments(vqvae_model, no_es_model, no_gated_model, data_loc, output_loc, conf):
    vqvae_model.eval()
    no_es_model.eval()
    no_gated_model.eval()
    vgg16 = models.vgg16(weights='IMAGENET1K_V1').features.to("cuda")
    data_loc = Path(data_loc)
    output_loc = Path(output_loc)
    nca_image_loc = output_loc / "nca_images"
    if not nca_image_loc.exists():
        nca_image_loc.mkdir()
    perturbed_image_loc = output_loc / "perturbed_images"
    if not perturbed_image_loc.exists():
        perturbed_image_loc.mkdir()
    our_image_loc = output_loc / "our_images"
    if not our_image_loc.exists():
        our_image_loc.mkdir()
    all_code_paths = list(data_loc.rglob("*.txt"))
    row_list = []
    clip_metrics = ClipMetrics()

    for code_path in tqdm(all_code_paths):
        img_name = code_path.stem
        output_folder = output_loc / img_name
        if not output_folder.exists():
            output_folder.mkdir()

        original_img = read_txt_and_decode_code(code_path, vqvae_model)
        utils.save_image(original_img, output_folder / "original.png", normalize=True)

        generated_images = []
        for idx in range(5):
            new_code_path = (code_path.parent / code_path.stem) / f"new_{idx}.txt.lvl"
            generated_img = read_txt_and_decode_code(new_code_path, vqvae_model)
            generated_images.append(generated_img)
            utils.save_image(generated_img, our_image_loc / f"generated_{idx}.png", normalize=True)

        # Train and generate images using NCA model
        nca_model = train_nca_model(original_img.to("cuda"), vgg16)
        nca_images = []
        for idx in range(5):
            nca_img = nca_model.generate_image(original_img.to("cuda"))
            nca_images.append(nca_img)
            utils.save_image(nca_img, nca_image_loc / f"nca_generated_{idx}.png", normalize=True)

        # Generate images by perturbing the latent and using our VQ-VAE model
        perturbed_images = []
        for idx in range(5):
            perturbed_img = read_txt_perturb_and_decode(code_path, vqvae_model)
            perturbed_images.append(perturbed_img)
            utils.save_image(perturbed_img, perturbed_image_loc / f"perturbed_{idx}.png", normalize=True)

        # Save comparison images
        comparison_image = torch.cat(generated_images + nca_images + perturbed_images, 1)
        utils.save_image(comparison_image, output_folder / "comparison.png", normalize=True)


        row_list.append({
            "filename": img_name,
            "real_class": None,
            "our_class": None,
            "nca_class": None,
            "perturbed_class": None,
        })

    our_tce = clip_metrics.tce(str(our_image_loc))
    nca_tce = clip_metrics.tce(str(nca_image_loc))
    perturbed_tce = clip_metrics.tce(str(perturbed_image_loc))

    print(f"Our TCE: {our_tce}")
    print(f"NCA TCE: {nca_tce}")
    print(f"Perturbed TCE: {perturbed_tce}")
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