from utils import prepare_data, display
from generative_model import *
from generative_tests import interpolate_ims, check_generative_recall, plot_history, vector_arithmetic, \
    plot_latent_space_with_labels
from tensorflow import keras
import numpy as np
from random import randrange
import hopfield_utils
import matplotlib.backends.backend_pdf
from config import dims_dict
from utils import prepare_data
from tqdm import tqdm
import os
import pickle


def run_end_to_end(dataset='mnist', generative_epochs=10, num=100, latent_dim=5, kl_weighting=1,
                   hopfield_type='continuous', lr=0.001, do_vector_arithmetic=False, interpolate=False, plot_space=True,
                   hopfield_beta=100, use_weights=True):

    # Paths to write results to:
    pdf_path = "./outputs/output_{}_{}items_{}eps_{}lv_{}lr_{}kl.pdf".format(dataset,
                                                                             num,
                                                                             generative_epochs,
                                                                             latent_dim,
                                                                             lr,
                                                                             kl_weighting)
    history_path = "./outputs/history_{}_{}items_{}eps_{}lv_{}lr_{}kl.pkl".format(dataset,
                                                                             num,
                                                                             generative_epochs,
                                                                             latent_dim,
                                                                             lr,
                                                                             kl_weighting)
    decoding_path = "./outputs/decoding_{}_{}items_{}eps_{}lv_{}lr_{}kl.pkl".format(dataset,
                                                                             num,
                                                                             generative_epochs,
                                                                             latent_dim,
                                                                             lr,
                                                                             kl_weighting)

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    print("Preparing datasets")
    train_data, test_data, noisy_train_data, noisy_test_data, train_labels, test_labels = prepare_data(dataset,
                                                                                                       labels=True)

    dims = dims_dict[dataset]

    # Load vae for dataset from model_weights directory is use_weights is set to True
    if use_weights is True:
        print("Using existing weights - set use_weights to False to train a new model with the specified parameters")
        encoder, decoder = models_dict[dataset](latent_dim=latent_dim)
        encoder.load_weights("model_weights/{}_encoder.h5".format(dataset))
        decoder.load_weights("model_weights/{}_decoder.h5".format(dataset))
        vae = VAE(encoder, decoder, kl_weighting=1)
        net = None

    else:
        # Sampling from the MHN takes a while, so use previous run's 'replayed memories' if available
        # Just delete the 'predictions_*.npy' files to regenerate
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
        # Build VAE (large or small version depending on dataset) with latent_dim latent variables
        encoder, decoder = models_dict[dataset](latent_dim=latent_dim)
        vae = VAE(encoder, decoder, kl_weighting)
        # jit_compile set to False to run on MacOS
        opt = keras.optimizers.Adam(lr=lr, amsgrad=True, jit_compile=False)
        vae.compile(optimizer=opt)
        decoding_history = DecodingHistory(dataset)
        history = tf.keras.callbacks.History()
        # Stop training if no loss improvement for three epochs
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
            # Select a random class
            random_class = np.random.choice(range(len(set(train_labels))))
            # Find the indices of samples belonging to the selected class
            class_indices = np.where(test_labels == random_class)[0]
            # Randomly select two indices from the same class
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

        
    