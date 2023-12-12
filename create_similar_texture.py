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
import uuid
import pickle


def load_texture(texture_path, conf):

    img = torchvision.io.read_image(str(texture_path), mode =torchvision.io.ImageReadMode.RGB)
    # Convert from uint8 to float32
    img = img.float()
    img = img.cuda()

    # Normalize to mean 0.5 stdv 0.5 i guess
    normalizer = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    resize = torchvision.transforms.Resize(conf.data.size)

    # Add a batch dimension
    img = img[None, :, :, :]

    img = resize(img)
    img = normalizer(img)
    return img

def get_texture_codes(texture_tensor, model):

    quant_t, quant_b, diff, id_t, id_b = model.encode(texture_tensor)

    return quant_t, quant_b, id_t, id_b

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


def run_wfc_generation(trained_wfc_model, width_height, iteration_levels = 2, wrapping = False):

    level_height = width_height[1]
    level_width = width_height[0]

    new_texture_codes = generate_new_level(level_height, level_width, trained_wfc_model,
                               wrapping=wrapping, max_attempts=5, iteration_levels = iteration_levels)

    return new_texture_codes


def decode_latents(id_t, id_b, model):

    decoded = model.decode_code(id_t, id_b)

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


if __name__ == "__main__":

    # read args and conf

    texture_path, model_path, output_path = __parse_args()
    output_path = make_folder_structure(output_path)
    conf = __load_config()

    # Load texture and model

    texture_tensor = load_texture(texture_path, conf)
    model = load_model(model_path)

    # Get texture codebook idxs

    # TODO: test, are these the right shape?
    quant_t, quant_b, id_t, id_b = get_texture_codes(texture_tensor, model)

    # Run WFC on texture embedding

    wfc_model_b = train_texture_wfc(texture_codes=id_b.cpu().numpy(), window_size=2, wrapping=False)
    new_id_b = run_wfc_generation(wfc_model_b, width_height=(32, 32), iteration_levels=2, wrapping=False)

    wfc_model_t = train_texture_wfc(texture_codes=id_t.cpu().numpy(), window_size=2, wrapping=False)
    new_id_t = run_wfc_generation(wfc_model_t, width_height=(16, 16), iteration_levels=2, wrapping=False)


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
    torchvision.utils.save_image(torch.cat([new_textures_b, new_textures_t, fully_new_textures], 0), img_output_path, normalize=True)


    # Plot

    pass

# Below is the old tensorflow-based version.
'''
def load_texture(texture_path):
    img = tf.io.read_file(texture_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32).numpy()
    texture = tf.image.rgb_to_hsv(img)

    return img,texture

def get_texture_codes(texture, vqvae):
    encoder = vqvae.get_layer("encoder")
    quantizer = vqvae.get_layer("vector_quantizer")

    encoded_texture = encoder.predict(np.expand_dims(texture, 0)) # Expanding dims to have batch size as well
    flat_encoded_texture = encoded_texture.reshape(-1, encoded_texture.shape[-1])
    code_indices = quantizer.get_code_indices(flat_encoded_texture)
    code_indices = code_indices.numpy().reshape(encoded_texture.shape[:-1])

    return code_indices


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

def run_wfc_generation(trained_wfc_model, width_height, iteration_levels = 2, wrapping = False):

    level_height = width_height[1]
    level_width = width_height[0]

    new_texture_codes = generate_new_level(level_height, level_width, trained_wfc_model,
                               wrapping=wrapping, max_attempts=5, iteration_levels = iteration_levels)

    return new_texture_codes

def load_model(vqvae_path):
    vqvae = keras.models.load_model(vqvae_path, custom_objects={"VectorQuantizer": VectorQuantizer})

    return vqvae
def save_WFC(WFC_Variable, save_path):
    with open(str(save_path), 'wb') as f:
        pickle.dump(WFC_Variable, f)

def load_WFC(save_path):
    with open(str(save_path), 'rb') as file:
        new_texture_codes = pickle.load(file)
    return new_texture_codes

def __parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--texture_loc")
    parser.add_argument("--vqvae_loc")
    parser.add_argument("--save_loc")

    args = parser.parse_args()

    return Path(args.texture_loc), Path(args.vqvae_loc), Path(args.save_loc)

def run_decoder(texture_codes, vqvae, num_embeddings, encoding_shape):

    quantizer = vqvae.get_layer("vector_quantizer")
    decoder = vqvae.get_layer("decoder")
    pretrained_embeddings = quantizer.embeddings
    # Convert the code indices to one-hot encoded vectors
    codes_onehot = tf.one_hot(texture_codes, num_embeddings, axis=-1)
    # Use the one-hot encoded vectors to index into the pretrained embeddings
    quantized = tf.matmul(codes_onehot, pretrained_embeddings, transpose_b=True)
    # Reshape the quantized vectors to the correct shape
    quantized = tf.reshape(quantized, (-1, *(encoding_shape)))
    # Run the decoder on the quantized vectors to generate the new texture
    new_texture = decoder.predict(quantized)

    return new_texture

if __name__ == "__main__":

    num_new_textures = 3
    NUM_EMBEDDINGS = 32
    LATENT_DIM = 32
    LATENT_WIDTH_HEIGHT = (64, 64)
    ITERATION_LEVELS = 8
    WINDOW_SIZE = 2
    WRAPPING = True
    MODEL_SAVED = False

    texture_path, vqvae_path, save_path = __parse_args()

    model = load_model(vqvae_path)
    print("Loaded model")

    # This code chunk was used to debug the vq-vae and should be moved to its own file if useful.

    texture, normalized_texture = load_texture(str(texture_path))

    texture_codes = get_texture_codes(normalized_texture, model)
    print("Training WFC...")
    texture_wfc = train_texture_wfc(texture_codes, WINDOW_SIZE, WRAPPING)
    filename = str(uuid.uuid4())

    for idx in range(num_new_textures):
        print(f"Generating new texture number {idx + 1}...")

        codes_save_loc = save_path / f'{texture_path.stem}_generated_codes_{idx}.pickle'

        if(not MODEL_SAVED):
            new_texture_codes = run_wfc_generation(texture_wfc, LATENT_WIDTH_HEIGHT, iteration_levels=ITERATION_LEVELS)
            save_WFC(new_texture_codes, codes_save_loc)
        else:
            new_texture_codes = load_WFC(codes_save_loc)

        print("Codes generated, decoding...")
        # Last parameter is the embedding size, it should match the VQVAE embedding size
        new_texture = run_decoder(new_texture_codes, model, num_embeddings=NUM_EMBEDDINGS,
                                   encoding_shape=(LATENT_WIDTH_HEIGHT[0], LATENT_WIDTH_HEIGHT[1], LATENT_DIM))

        gen_texture = new_texture[0]
        gen_texture = tf.image.hsv_to_rgb(gen_texture)
        img_save_path = save_path / f'texture_{filename}_LD{LATENT_DIM}_NE{NUM_EMBEDDINGS}_WS{WINDOW_SIZE}_{idx}.png'
        plot_results1(tf.keras.preprocessing.image.array_to_img(texture), new_texture_codes, tf.keras.preprocessing.image.array_to_img(gen_texture), img_save_path)

    print("Enjoy!")
    
'''

