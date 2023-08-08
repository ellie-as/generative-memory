import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class Sampling(layers.Layer):
    # Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder_decoder(vectorizer, latent_dim=100, l1_value=0.01):
    input_dim = len(vectorizer.get_feature_names_out())
    encoder_inputs = keras.Input(shape=(input_dim,))
    dropped_out = layers.Dropout(0.5, name="dropout_layer")(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(dropped_out)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_inputs)
    # This uses the special sampling layer defined above:
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    decoder_outputs = layers.Dense(input_dim, activation='sigmoid',
                                   activity_regularizer=regularizers.l1(l1_value))(latent_inputs)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


def build_encoder_decoder_backup(vectorizer, latent_dim=100, l1_value=0.01):
    input_dim = len(vectorizer.get_feature_names_out())
    encoder_inputs = keras.Input(shape=(input_dim,))
    dropped_out = layers.Dropout(0.05, name="dropout_layer")(encoder_inputs)
    dropped_out = layers.Dense(input_dim * 0.1)(dropped_out)
    z_mean = layers.Dense(latent_dim, name="z_mean")(dropped_out)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_inputs)
    # This uses the special sampling layer defined above:
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    decoder_outputs = layers.Dense(input_dim * 0.1)(latent_inputs)
    decoder_outputs = layers.Dense(input_dim, activation='sigmoid',
                                   activity_regularizer=regularizers.l1(l1_value))(decoder_outputs)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            #reconstruction_loss *= input_dim
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + self.beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def train_vae(x_train, vectorizer, eps=1000, ld=200, beta=0.0001, batch=2, model_save_path='model.h5',
              l1_value=0.01):
    encoder, decoder = build_encoder_decoder(vectorizer, latent_dim=ld, l1_value=l1_value)
    vae = VAE(encoder, decoder, beta)
    opt = keras.optimizers.Adam(amsgrad=True, jit_compile=False)
    vae.compile(optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='min')
    vae.fit(x_train, epochs=eps, batch_size=batch, verbose=True, shuffle=True, callbacks=[early_stopping])
    vae.save_weights(model_save_path)
    return vae


def load_vae(vectorizer, ld=200, beta=0.0001, model_weights_path='model.h5'):
    encoder, decoder = build_encoder_decoder(vectorizer, latent_dim=ld)
    vae = VAE(encoder, decoder, beta)

    # Create a dummy batch (e.g., zeros), and call it on your model:
    dummy_data = tf.zeros((1, len(vectorizer.vocabulary_)))
    _ = vae(dummy_data)

    vae.load_weights(model_weights_path)
    return vae
