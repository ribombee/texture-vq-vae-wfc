from pathlib import Path
from model import VectorQuantizer
import cv2 as cv
import tensorflow.keras as keras
import numpy as np
from util import plot_results
from WFC_train import extract_patterns, compute_pattern_occurrences, get_unique_patterns, compute_adjacencies
from WFC_generate import generate_new_level


def load_texture(texture_path):
    texture = cv.imread(str(texture_path))
    texture = cv.cvtColor(texture, cv.COLOR_BGR2HSV)

    return texture

def get_texture_codes(texture, vqvae):
    encoder = vqvae.get_layer("encoder")
    quantizer = vqvae.get_layer("vector_quantizer")

    encoded_texture = encoder.predict(np.expand_dims(texture, 0)) # Expanding dims to have batch size as well
    flat_encoded_texture = encoded_texture.reshape(-1, encoded_texture.shape[-1])
    code_indices = quantizer.get_code_indices(flat_encoded_texture)
    code_indices = code_indices.numpy().reshape(encoded_texture.shape[:-1])

    return code_indices


def train_texture_wfc(texture_codes):

    wrapping = True
    pattern_height = 2
    pattern_width = 2
    row_offset = 1
    col_offset = 1

    all_patterns = extract_patterns([texture_codes], pattern_height, pattern_width,
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

def run_wfc_generation(trained_wfc_model):

    wrapping = True
    level_height = 32
    level_width = 32

    new_texture_codes = generate_new_level(level_height, level_width, trained_wfc_model,
                               wrapping=wrapping, max_attempts=20)

    return new_texture_codes

def load_model(vqvae_path):
    vqvae = keras.models.load_model(vqvae_path, custom_objects={"VectorQuantizer": VectorQuantizer})

    return vqvae

def __parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--texture_loc")
    parser.add_argument("--vqvae_loc")
    parser.add_argument("--save_loc")

    args = parser.parse_args()

    return Path(args.texture_loc), Path(args.vqvae_loc), Path(args.save_loc)

def run_decoder(texture_codes, num_new_patterns, decoder):

    for idx in range(num_new_patterns):

        new_texture = decoder.predict()
    pass

if __name__ == "__main__":

    num_new_textures = 5

    texture_path, vqvae_path, save_path = __parse_args()

    model = load_model(vqvae_path)

    texture = load_texture(texture_path)

    texture_codes = get_texture_codes(texture, model)

    texture_wfc = train_texture_wfc(texture_codes)

    new_texture_codes = run_wfc_generation(texture_wfc)

    new_textures = run_decoder(new_texture_codes, num_new_textures, model.get_layer("decoder"))

    for idx in range(num_new_textures):
        plot_results(texture, new_texture_codes[idx], new_textures[idx])