import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


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

    def generate_image(self, original_img, steps=100):
        x = self.seed(1, original_img.shape[-1]).cuda()
        for _ in range(steps):
            x = self(x)
        return to_rgb(x)

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
  yy = calc_styles_vgg(target_img.unsqueeze(0), vgg16)
  def loss_f(imgs):
    xx = calc_styles_vgg(imgs, vgg16)
    return sum(ot_loss(x, y) for x, y in zip(xx, yy))
  return loss_f

def to_nchw(img):
  img = torch.as_tensor(img)
  if len(img.shape) == 3:
    img = img[None,...]
  return img.permute(0, 3, 1, 2)

def project_sort(x, proj):
    return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

def ot_loss(source, target, proj_n=32):
    ch, n = source.shape[-2:]
    projs = F.normalize(torch.randn(ch, proj_n), dim=0)
    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)
    target_interp = F.interpolate(target_proj, n, mode='nearest')
    return (source_proj - target_interp).square().sum()

def train_nca_model(original_img, vgg16, num_steps=5000):
    ca = CA()
    opt = torch.optim.Adam(ca.parameters(), 1e-3, capturable=True)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1000, 2000], 0.3)
    loss_log = []
    with torch.no_grad():
        pool = ca.seed(256)
        # We do not need to use to_nchw here because the image is already in NCHW format
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
        with torch.no_grad():
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