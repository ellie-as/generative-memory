import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow import keras
from utils import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


def label_classifier(latents, labels, num=200):
    np.random.seed(1)
    x_train, x_test, y_train, y_test = train_test_split(latents[0], labels,
                                                        test_size=0.5, random_state=1)
    clf = make_pipeline(StandardScaler(), SVC())
    clf.fit(x_train[0:num], y_train[0:num])
    score = clf.score(x_test, y_test)
    return score


class DecodingHistory(keras.callbacks.Callback):

    def __init__(self, dataset):
        _, self.test_data, _, _, _, self.test_labels = prepare_data(dataset, labels=True)
        self.decoding_history = []

    def on_epoch_begin(self, epoch, logs=None):
        latents = self.model.encoder.predict(self.test_data)
        score = label_classifier(latents, self.test_labels)
        self.decoding_history.append(score)


class Sampling(layers.Layer):
    # Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def encoder_network_large(input_shape, latent_dim=100):
    input_img = layers.Input(shape=input_shape)
    x = layers.Dropout(0.2, input_shape=input_shape)(input_img)
    x = layers.Conv2D(32, 4, strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 4, strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 4, strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    z_mean = layers.Dense(latent_dim, name='mean')(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(input_img, [z_mean, z_log_var, z], name="encoder")
    return encoder, z_mean, z_log_var


def decoder_network_large(latent_dim=100):
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(4096)(decoder_input)
    x = layers.Reshape((4, 4, 256))(x)

    x = layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(128, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(32, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D((2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(3, 3, strides=1, padding='same', activation='sigmoid')(x)

    decoder = keras.Model(decoder_input, x)
    return decoder


def build_encoder_decoder_large(latent_dim=5):
    input_shape = (64, 64, 3)
    encoder, z_mean, z_log_var = encoder_network_large(input_shape, latent_dim)
    decoder = decoder_network_large(latent_dim)
    return encoder, decoder


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


models_dict = {"shapes3d": build_encoder_decoder_large}
    