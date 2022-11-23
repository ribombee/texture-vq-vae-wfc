from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Much of this code is from https://keras.io/examples/generative/vq_vae/


def ssim_loss(y_true, y_pred):

    reconstruction_ssim_loss = 0
    reconstruction_ssim_loss += (1 - tf.reduce_mean(tf.image.ssim(tf.image.rgb_to_yuv(tf.image.hsv_to_rgb(y_true)),
                                                                  tf.image.rgb_to_yuv(tf.image.hsv_to_rgb(y_pred)), max_val=1.0)))

    return reconstruction_ssim_loss

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
                "beta": self.beta,
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


def get_encoder(latent_dim=16, latent_shrink_scale=2):
    encoder_inputs = keras.Input(shape=(64, 64, 3))
    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2D(32, 3, activation="relu", strides=strides, padding="same")(
        encoder_inputs
    )

    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2D(64, 3, activation="relu", strides=strides, padding="same")(x)

    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2D(64, 3, activation="relu", strides=strides, padding="same")(x)

    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2D(64, 3, activation="relu", strides=strides, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16, latent_shrink_scale=2):
    latent_inputs = keras.Input(shape=get_encoder(latent_dim, latent_shrink_scale).output.shape[1:])

    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=strides, padding="same")(
        latent_inputs
    )

    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=strides, padding="same")(x)

    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=strides, padding="same")(x)

    if latent_shrink_scale > 0:
        strides = 2
        latent_shrink_scale -= 1
    else:
        strides = 1

    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=strides, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=strides, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def get_vqvae(latent_dim=16, num_embeddings=64, latent_shrink_scale=2):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim, latent_shrink_scale)
    decoder = get_decoder(latent_dim, latent_shrink_scale)
    inputs = keras.Input(shape=(64, 64, 3))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, latent_shrink_scale=2, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.latent_shrink_scale = latent_shrink_scale

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings, self.latent_shrink_scale)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.ssim_loss_tracker = keras.metrics.Mean(name="ssim_loss")

    def call(self, x, training=None, mask=None):
        return self.vqvae.call(x)

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
            reconstruction_ssim_loss = ssim_loss(x, reconstructions)
            total_loss = reconstruction_loss + sum(self.vqvae.losses) + reconstruction_ssim_loss

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.ssim_loss_tracker.update_state(reconstruction_ssim_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
            "ssim_loss": self.ssim_loss_tracker.result()
        }

    def test_step(self, x):
        # The exact same thing as train_step
        return self.train_step(x)
