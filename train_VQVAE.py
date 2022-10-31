import random

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow as tf
import cv2 as cv
from datetime import datetime
from pathlib import Path
from model import *


def get_image_processor(image_size, augment=False):

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2, 0.2),
        keras.layers.RandomCrop(image_size[0], image_size[1])]
    )

    def process_image(image):
        if augment:
            if np.random.rand() > 0.7:
                image = data_augmentation(image)

        image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])

        image = image / 255 # Normalize
        image = tf.image.rgb_to_hsv(image) # Swap to hsv

        return image

    return process_image

def load_dtd_dataset(image_size, batch_size):
    train_ds, val_ds, test_ds = tfds.load('dtd', split=['train', 'validation', 'test'], shuffle_files=True)

    train_ds = train_ds.map(lambda items: (items["image"]))
    val_ds = val_ds.map(lambda items: (items["image"]))

    train_image_processor = get_image_processor(image_size, augment=True)
    val_test_image_processor = get_image_processor(image_size, augment=False)

    train_ds = (train_ds.cache().map(train_image_processor).batch(batch_size))
    val_ds = (val_ds.cache().map(val_test_image_processor).batch(batch_size))

    return train_ds, val_ds

def plot_results(original, codes, reconstruction, img_save_path):
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(original, cv.COLOR_HSV2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(codes)
    plt.title("Codes")
    plt.axis("Off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(reconstruction, cv.COLOR_HSV2RGB))
    plt.title("Reconstruction")
    plt.axis("off")

    plt.savefig((img_save_path / f"{idx}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def __parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # If training new model, save in new folder in this location.
    # If not training, load model saved in this folder. Will not look deeper.
    parser.add_argument("--model_loc")
    args = parser.parse_args()

    return Path(args.model_loc)

if __name__ == "__main__":
    print("Starting")

    # SETTINGS
    TRAIN_NEW_MODEL = True
    BATCH_SIZE = 16
    IMAGE_SIZE = (64, 64)   # To match DOOM texture size
    EPOCHS = 500
    LATENT_DIM = 16
    NUM_EMBEDDINGS = 128
    # How much smaller the latent representation should be. Scaled as a multiple of 2^(-LATENT_SHRINK_SCALE)
    LATENT_SHRINK_SCALE = 1

    # Parse command line arguments
    model_path = __parse_args()

    # GET DATASET
    # TODO: Add data augmentation to dataset
    train_ds, val_ds = load_dtd_dataset(IMAGE_SIZE, BATCH_SIZE)

    if TRAIN_NEW_MODEL:
        vars = []
        for batch in train_ds:
            batch_variance = tfp.stats.variance(batch)
            vars.append(batch_variance)

        # TODO: calculate actual variance!!
        # NOTE this is not the variance of the entire dataset. See https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches and update later.
        data_variance = np.mean(vars)

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10)

        vqvae_trainer = VQVAETrainer(data_variance, latent_dim=LATENT_DIM,
                                     num_embeddings=NUM_EMBEDDINGS,
                                     latent_shrink_scale=LATENT_SHRINK_SCALE)
        print(vqvae_trainer.vqvae.summary())
        vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
        vqvae_trainer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])
        vqvae = vqvae_trainer.vqvae

        results = vqvae_trainer.evaluate(val_ds)
        time_now = datetime.now().strftime('%m-%d-%H-%m')

        print("Model trained. Saving model.")

        keras.models.save_model(vqvae_trainer.vqvae,
                                model_path / f'vqvae_{time_now}_{str(results)}')
        # keras.models.save_model(vqvae_trainer, model_path / 'trainer_{time_now}_{str(results)}')
    else:
        vqvae = keras.models.load_model(model_path, custom_objects={"VectorQuantizer": VectorQuantizer})

    print(vqvae.summary())

    val_batch = list(val_ds.take(1).as_numpy_iterator())
    encoder = vqvae.get_layer("encoder")
    quantizer = vqvae.get_layer("vector_quantizer")
    decoder = vqvae.get_layer("decoder")

    input_batch = val_batch[0]

    encoded_batch = encoder.predict(input_batch)
    flat_encoded_batch = encoded_batch.reshape(-1, encoded_batch.shape[-1])
    code_batch_indices = quantizer.get_code_indices(flat_encoded_batch)
    code_batch_indices = code_batch_indices.numpy().reshape(encoded_batch.shape[:-1])
    quantized_batch = quantizer(encoded_batch)
    output_batch = decoder.predict(quantized_batch)

    if not TRAIN_NEW_MODEL:
        img_save_path = model_path / "val_plots"
    else:
        img_save_path = model_path / f'vqvae_{time_now}_{str(results)}' / "val_plots"

    if not img_save_path.exists():
        img_save_path.mkdir()

    for idx in range(input_batch.shape[0]):

        plot_results(input_batch[idx], code_batch_indices[idx], output_batch[idx], img_save_path)

