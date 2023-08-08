import matplotlib.backends.backend_pdf
import numpy as np
import os
import pickle
from random import randrange
from tensorflow import keras
from tqdm import tqdm

import hopfield_utils
from config import dims_dict
from generative_model import *
from generative_tests import interpolate_ims, check_generative_recall, plot_history, vector_arithmetic, \
    plot_latent_space_with_labels
from utils import prepare_data, display, get_output_paths


def run_end_to_end(dataset='shapes3d', generative_epochs=10, num=100, latent_dim=5, kl_weighting=1,
                   hopfield_type='continuous', lr=0.001, do_vector_arithmetic=False, interpolate=False, plot_space=True,
                   hopfield_beta=100, use_weights=True):
    """
    Runs an end-to-end simulation of consolidation as teacher-student training of a generative network.

    Args:
        dataset (str, optional): Name of the dataset to use. Defaults to 'shapes3d'.
        generative_epochs (int, optional): Number of generative epochs. Defaults to 10.
        num (int, optional): Number of items. Defaults to 100.
        latent_dim (int, optional): Dimension of the latent variables. Defaults to 5.
        kl_weighting (int, optional): Weighting for the Kullback-Leibler divergence. Defaults to 1.
        hopfield_type (str, optional): Type of the Hopfield network. Defaults to 'continuous'.
        lr (float, optional): Learning rate. Defaults to 0.001.
        do_vector_arithmetic (bool, optional): Whether to perform vector arithmetic. Defaults to False.
        interpolate (bool, optional): Whether to perform interpolation. Defaults to False.
        plot_space (bool, optional): Whether to plot the resulting latent space. Defaults to True.
        hopfield_beta (int, optional): Beta value for the modern Hopfield network. Defaults to 100.
        use_weights (bool, optional): Whether to use saved model weights. Defaults to True.

    Returns:
        net: The modern Hopfield network representing the initial hippocampal network
        vae: The trained variational autoencoder representing the neocortical generative network
    """

    # get paths to write results to
    pdf_path, history_path, decoding_path = get_output_paths(dataset, num, generative_epochs, latent_dim, lr,
                                                             kl_weighting)

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    print("Preparing datasets")
    train_data, test_data, noisy_train_data, noisy_test_data, train_labels, test_labels = prepare_data(dataset,
                                                                                                       labels=True)

    dims = dims_dict[dataset]

    # load vae for dataset from model_weights directory is use_weights is set to True
    if use_weights is True:
        print("Using existing weights - set use_weights to False to train a new model with the specified parameters")
        encoder, decoder = models_dict[dataset](latent_dim=latent_dim)
        encoder.load_weights("model_weights/{}_encoder.h5".format(dataset))
        decoder.load_weights("model_weights/{}_decoder.h5".format(dataset))
        vae = VAE(encoder, decoder, kl_weighting=1)
        net = None

    else:
        # sampling from the MHN takes a while, so use previous run's 'replayed memories' if available
        # just delete the 'predictions_*.npy' files to regenerate
        np_f = './mhn_memories/predictions_{}_{}.npy'.format(dataset, num)
        if os.path.exists(np_f):
            print("Using saved MHN predictions from previous run.")
            with open(np_f, 'rb') as fh:
                predictions = np.load(fh)
            net = None

        else:
            print("Creating Hopfield network.")
            net = hopfield_utils.create_hopfield(num, hopfield_type=hopfield_type, dataset=dataset, beta=hopfield_beta)
            predictions = []
            tests = []

            images_masked_np = hopfield_utils.mask_image_random(num)
            images_masked_np = [im_np.reshape(-1, 1) for im_np in images_masked_np]

            print("Sampling from modern Hopfield network.")
            for test_ind in tqdm(range(num)):
                test = images_masked_np[test_ind].reshape(-1, 1)
                if hopfield_type == 'classical':
                    reconstructed = net.retrieve(test)
                else:
                    reconstructed = net.retrieve(test, max_iter=10)
                predictions.append(reconstructed.reshape((1, dims[0], dims[1], dims[2])))
                tests.append(test.reshape((1, dims[0], dims[1], dims[2])))

            predictions = np.concatenate(predictions, axis=0)
            tests = np.concatenate(tests, axis=0)

            fig = display(tests, predictions, title='Inputs and outputs for modern Hopfield network')
            pdf.savefig(fig)
            # rescale predictions back to interval [0, 1]
            predictions = (predictions + 1) / 2

            with open('./mhn_memories/predictions_{}_{}.npy'.format(dataset, num), 'wb') as fh:
                np.save(fh, predictions)

        print("Starting to train VAE.")
        # build VAE with latent_dim latent variables
        encoder, decoder = build_encoder_decoder_large(latent_dim=latent_dim)
        vae = VAE(encoder, decoder, kl_weighting)
        # jit_compile set to False to run on MacOS
        opt = keras.optimizers.Adam(lr=lr, amsgrad=True, jit_compile=False)
        vae.compile(optimizer=opt)
        decoding_history = DecodingHistory(dataset)
        history = tf.keras.callbacks.History()
        # stop training if no loss improvement for three epochs
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        print("Input data shape:", predictions.shape)
        vae.fit(predictions, epochs=generative_epochs, verbose=2, batch_size=32, shuffle=True,
                callbacks=[history, decoding_history, early_stopping])

        vae.encoder.save_weights("model_weights/{}_encoder.h5".format(dataset))
        vae.decoder.save_weights("model_weights/{}_decoder.h5".format(dataset))

        fig = plot_history(history, decoding_history)
        pdf.savefig(fig, bbox_inches='tight')

        pickle.dump(history.history['reconstruction_loss'], open(history_path, "wb"))
        pickle.dump(decoding_history.decoding_history, open(decoding_path, "wb"))

    print("Recalling noisy images with the generative model:")
    fig = check_generative_recall(vae, train_data[0:100])
    pdf.savefig(fig)

    latents = vae.encoder.predict(test_data)

    if interpolate is True:
        print("Interpolating between image pairs:")
        for i in range(10):
            fig = interpolate_ims(latents, vae, randrange(50), randrange(50))
            pdf.savefig(fig)

    if do_vector_arithmetic is True:
        print("Doing vector arithmetic:")
        for i in range(10):
            # select a random class
            random_class = np.random.choice(range(len(set(train_labels))))
            # find the indices of samples belonging to the selected class
            class_indices = np.where(test_labels == random_class)[0]
            # randomly select two indices from the same class
            first, third = np.random.choice(class_indices, size=2, replace=False)
            second = randrange(100)
            print(first, second, third)
            fig = vector_arithmetic(test_data, latents, vae, first, second, third)
            pdf.savefig(fig)

    if plot_space is True:
        print("Plotting latent space with labels:")
        fig = plot_latent_space_with_labels(latents, test_labels)
        pdf.savefig(fig)

    pdf.close()

    return net, vae
