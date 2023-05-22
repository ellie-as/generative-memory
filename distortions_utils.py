import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from utils import display
import matplotlib.pyplot as plt
import numpy as np


def kl_loss_fn(z_mean, z_log_var, kl_weighting):
    # take the sum across the n latent variables
    # then take the mean across the batch
    kl = K.mean(-0.5 * K.sum(1 + z_log_var \
                      - K.square(z_mean) \
                      - K.exp(z_log_var), axis=-1))
    return kl_weighting * kl


def reconstruction_loss_fn(x, t_decoded):
    # mean_absolute_error() returns result of dim (n_in_batch, pixels)
    # take the sum across the 64x64x3 pixels
    # take the mean across the batch
    data = x
    reconstruction = t_decoded
    # note that binary_crossentropy loss also gives good results
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
        keras.losses.mean_absolute_error(data, reconstruction), axis=(1, 2)))
    return reconstruction_loss


class Sampling(layers.Layer):
    # Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder_decoder_small(latent_dim=10):
    # Encoder
    input_layer = layers.Input(shape=(28, 28, 1))
    x = layers.Dropout(0.1, input_shape=(28, 28, 1))(input_layer)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(latent_dim)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


class VAE(keras.Model):

    def __init__(self, encoder, decoder, kl_weighting=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weighting = kl_weighting
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = reconstruction_loss_fn(data, reconstruction)
            kl_loss = kl_loss_fn(z_mean, z_log_var, self.kl_weighting)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def display(array1, array2, seed=None, title='Inputs and outputs of the model', n=10):
    hopfield = False

    dim = array1[0].shape[0]
    # Displays ten random images from each one of the supplied arrays.
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    fig = plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        if hopfield is True:
            plt.imshow(image1.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image1.reshape(dim, dim))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        if hopfield is True:
            plt.imshow(image2.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image2.reshape(dim, dim))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle(title)
    plt.show()
    return fig


def check_generative_recall(vae, test_data, test_labels, latent_umap, pixel_umap, displaybool=False, n=2000):
    latents = vae.encoder.predict(test_data)
    predictions = vae.decoder.predict(latents[0])

    if displaybool is True:
        fig = display(test_data, predictions, seed=1)
        plot_latent_space_with_labels(latents, test_labels, latent_umap, titles=True, n=n)
        plot_pixel_space_with_labels(test_data, test_labels, pixel_umap, titles=True, n=n)
    return predictions, latents


def plot_latent_space_with_labels(latents, labels, umap, titles=False, n=2000):
    np.random.seed(1)
    fig = plt.figure(figsize=(4, 4))

    embedded = umap.transform(latents[0][0:n])
    x = [x[0] for x in embedded]
    y = [x[1] for x in embedded]

    plt.scatter(x, y, c=labels[0:n], alpha=0.5, cmap=plt.cm.plasma)
    if titles is True:
        plt.title('Latent space in 2D, colour-coded by label')
    plt.show()
    return fig


def plot_pixel_space_with_labels(test_data, labels, umap, titles=False, n=2000):
    np.random.seed(1)
    fig = plt.figure(figsize=(4, 4))

    embedded = umap.transform(test_data[0:n].reshape(n, 784))
    x = [x[0] for x in embedded]
    y = [x[1] for x in embedded]

    plt.scatter(x, y, c=labels[0:n], alpha=0.5, cmap=plt.cm.plasma)
    if titles is True:
        plt.title('Pixel space in 2D, colour-coded by label')
    plt.show()
    return fig