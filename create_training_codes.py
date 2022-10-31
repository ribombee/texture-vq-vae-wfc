from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from model import VectorQuantizer
import numpy as np
import tensorflow_datasets as tfds


def get_image_processor(image_size):

    def process_image(image):
        image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])

        image = image / 255 # Normalize
        image = tf.image.rgb_to_hsv(image) # Swap to hsv

        return image

    return process_image

def load_dtd_dataset(image_size, batch_size):
    train_ds, val_ds, test_ds = tfds.load('dtd', split=['train', 'validation', 'test'], shuffle_files=True)

    train_ds = train_ds.map(lambda items: (items["image"]))
    val_ds = val_ds.map(lambda items: (items["image"]))

    train_image_processor = get_image_processor(image_size)
    val_test_image_processor = get_image_processor(image_size)

    train_ds = (train_ds.cache().map(train_image_processor).batch(batch_size))
    val_ds = (val_ds.cache().map(val_test_image_processor).batch(batch_size))

    return train_ds, val_ds

def __parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser
    parser.add_argument("--vqvae_loc")
    parser.add_argument ("--code_save_loc")

    args = parser.parse_args()

    return Path(args.vqvae_loc), Path(args.code_save_loc)

if __name__ == "__main__":

    vqvae_path, code_save_path = __parse_args()

    vqvae = keras.models.load_model(vqvae_path, custom_objects={"VectorQuantizer": VectorQuantizer})

    train_ds, val_ds = load_dtd_dataset((64, 64), 16)
    encoder = vqvae.get_layer("encoder")
    quantizer = vqvae.get_layer("vector_quantizer")
    decoder = vqvae.get_layer("decoder")

    idx = 0
    for batch in train_ds:

        encoded_batch = encoder.predict(batch)
        flat_encoded_batch = encoded_batch.reshape(-1, encoded_batch.shape[-1])
        code_batch_indices = quantizer.get_code_indices(flat_encoded_batch)
        code_batch_indices = code_batch_indices.numpy().reshape(encoded_batch.shape[:-1])

        for texture in batch:

            for row in texture:
                for column in row:
                    # Build out string for training WFC
                    pass

