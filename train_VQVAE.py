import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow as tf
import cv2 as cv
from datetime import datetime
from pathlib import Path

# Much of this code is from https://keras.io/examples/generative/vq_vae/

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def get_config(self):
        config = super(VectorQuantizer, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_embeddings": self.num_embeddings,
                "beta": self.beta
            }
        )
        return config

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


def get_encoder(latent_dim=16):
    encoder_inputs = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16):
    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def get_vqvae(latent_dim=16, num_embeddings=64):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(64, 64, 3))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    def test_step(self, x):
        # The exact same thing as train_step
        return self.train_step(x)

    def call(self, x):
        return self.vqvae.call(x)

def get_image_processor(image_size):
    def process_image(image):
        image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])
        # image = cv.cvtColor(image.eval(session=tf.compat.v1.Session().as_default()), cv.COLOR_RGB2HSV)

        # image_H, image_S, image_V = image.transpose(2,0,1)

        # image_H = image_H / 360
        # image_V = image_V / 255
        # image_S = image_S / 255

        image = image / 255 # Normalize
        image = tf.image.rgb_to_hsv(image) # Swap to hsv

        return image

    return process_image

def load_dtd_dataset(image_size, batch_size):
    train_ds, val_ds, test_ds = tfds.load('dtd', split=['train', 'validation', 'test'], shuffle_files=True)

    train_ds = train_ds.map(lambda items: (items["image"]))
    val_ds = val_ds.map(lambda items: (items["image"]))

    image_processor = get_image_processor(image_size)

    train_ds = (train_ds.cache().map(image_processor).batch(batch_size))
    val_ds = (val_ds.cache().map(image_processor).batch(batch_size))

    return train_ds, val_ds


if __name__ == "__main__":
    print("Starting")

    # SETTINGS
    BATCH_SIZE = 16
    IMAGE_SIZE = (64, 64)   # To match DOOM texture size
    EPOCHS = 500

    # GET DATASET
    # TODO: Add data augmentation to dataset
    train_ds, val_ds = load_dtd_dataset(IMAGE_SIZE, BATCH_SIZE)

    vars = []
    for batch in train_ds:
        batch_variance = tfp.stats.variance(batch)
        vars.append(batch_variance)

    # TODO: calculate actual variance!!
    # NOTE this is not the variance of the entire dataset. See https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches and update later.
    data_variance = np.mean(vars)

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)

    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    vqvae_trainer.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])

    results = vqvae_trainer.evaluate(val_ds)
    time_now = datetime.now().strftime('%m-%d-%H')

    keras.models.save_model(vqvae_trainer, f'E:\Documents\projects\VQVAEWFC\saved_models\\trainer_{time_now}_{str(results)}')
    keras.models.save_model(vqvae_trainer.vqvae,
                            f'E:\Documents\projects\VQVAEWFC\saved_models\\raw_model{time_now}_{str(results)}')

