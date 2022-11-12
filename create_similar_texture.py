from pathlib import Path
from model import VectorQuantizer
import cv2 as cv
import tensorflow.keras as keras
import tensorflow as tf
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

    wrapping = False
    pattern_height = 2
    pattern_width = 2
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

def run_wfc_generation(trained_wfc_model):

    wrapping = False
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

def run_decoder(texture_codes, num_new_patterns, vqvae, num_embeddings, encoding_shape):

    quantizer = vqvae.get_layer("vector_quantizer")
    decoder = vqvae.get_layer("decoder")
    new_textures = []
    for idx in range(num_new_patterns):
        pretrained_embeddings = quantizer.embeddings
        # Calling .numpy() on this otherwise matmul won't take it
        codes_onehot = tf.one_hot(texture_codes, num_embeddings).numpy()
        quantized = tf.matmul(
            codes_onehot.astype("float32"), pretrained_embeddings, transpose_b=True
        ) # This puts the appropriate encoding vector corresponding to the id that has been one-hot encoded
        quantized = tf.reshape(quantized, (-1, *(encoding_shape)))
        new_texture = decoder.predict(quantized)
        new_textures.append(new_texture)

    return new_textures

if __name__ == "__main__":

    num_new_textures = 5
    NUM_EMBEDDINGS = 128

    texture_path, vqvae_path, save_path = __parse_args()

    model = load_model(vqvae_path)
    print("Loaded model")

    # This code chunk was used to debug the vq-vae and should be moved to its own file if useful.
    '''
    all_texture_paths = texture_path.glob('*')

    for sprite_path in all_texture_paths:
        print(sprite_path)
        sprite = cv.cvtColor(cv.imread(str(sprite_path)), cv.COLOR_BGR2HSV)

        hue, sat, val = cv.split(sprite)
        hue = hue / 179
        sat = sat / 255
        val = val / 255

        normalized_sprite = np.stack([hue, sat, val])
        normalized_sprite = np.transpose(normalized_sprite, (1, 2, 0))

        recreation = model.predict(np.expand_dims(normalized_sprite, 0))

        hue, sat, val = recreation[0][:,:,0], recreation[0][:,:,1], recreation[0][:,:,2]

        hue = np.clip(hue * 179, 0, 179)
        sat = np.clip(sat * 255, 0, 255)
        val = np.clip(val * 255, 0, 255)

        scaled_recreation = np.stack([hue, sat, val])
        scaled_recreation = np.transpose(scaled_recreation, (1, 2, 0))

        embeds = model.get_layer("encoder").predict(np.expand_dims(sprite, 0))
        flat_codes = embeds.reshape(-1, embeds.shape[-1])
        codes = model.get_layer("vector_quantizer").get_code_indices(flat_codes)
        codes = codes.numpy().reshape(embeds.shape[:-1])
        plot_results(sprite, codes[0], recreation[0], str(save_path / sprite_path.stem))
    '''

    texture = load_texture(texture_path)

    hue, sat, val = cv.split(texture)
    hue = hue / 179
    sat = sat / 255
    val = val / 255

    normalized_texture = np.stack([hue, sat, val])
    normalized_texture = np.transpose(normalized_texture, (1, 2, 0))

    texture_codes = get_texture_codes(normalized_texture, model)
    print("Training WFC...")
    texture_wfc = train_texture_wfc(texture_codes)



    for idx in range(num_new_textures):
        print("...WFC trained, generating...")
        new_texture_codes = run_wfc_generation(texture_wfc)
        print("Codes generated, decoding...")
        # Last parameter is the embedding size, it should match the VQVAE embedding size
        new_texture = run_decoder(new_texture_codes, num_new_textures, model, num_embeddings=NUM_EMBEDDINGS,
                                   encoding_shape=(32, 32, 16))

        if num_new_textures == 1:
            # To match the shape when num_new_textures > 1
            new_textures = [new_textures]
            new_texture_codes = [new_texture_codes]

        img_save_path = save_path / f'texture_{idx}.png'
        plot_results(texture, new_texture_codes, new_texture[0][0], img_save_path)

    print("Enjoy!")
