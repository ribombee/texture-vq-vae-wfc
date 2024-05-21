import torch
from torch import nn

# TODO: a custom loss function that operates on the embedded latens of the VQ-VAE
# It takes in latens Y at resolution (B, C, W, H, L)
# It calculates the encoded non-quantized latent Y_l
# performs a gaussian blur, then a kernel-based sharpening to produce Y_lng
# The loss then quantizes both Y_l and Y_lng and returns MSE(Y_lq, Y_lngq)
# TODO: does the loss need to apply to the quantization step as well?
# note: quantization in VQ-VAE2 is cluster-based and not learned, but the quantization changes every so often.
# Does this make a loss like this redundant?

# Alternatively, if the goal is that the have latents only calculated based on local information,
# it may be possible to augment Y to create Y_h where Y has random holes put into it, and constrain it such that
# where Y is untouched, Y_lq and Y_hlq are the same. Or even Y_l and Y_hl to exclude any quantizer issues.
# Maybe just do a thing where you punish number of quantized latents per image.
#
def nongranularity_loss(y, y_true):
    # Gaussian blur, then sharpen, then quantize then compare
    pass