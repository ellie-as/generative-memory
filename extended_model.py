from extended_model_utils import *
from end_to_end import *
import tensorflow as tf
from hopfield_models import ContinuousHopfield
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
import logging


plt.rcParams["figure.figsize"] = (15, 3)
# set tensorflow random seed to make outputs reproducible
tf.random.set_seed(1)


def get_predictions(input_data, vae):
    latents = vae.encoder.predict(input_data)
    predictions = vae.decoder.predict(latents[0])
    return predictions


def find_min_max(arr):
    min_val = np.amin(arr)
    max_val = np.amax(arr)
    return min_val, max_val


def ims_to_hpc_format(input_ims, vae, threshold, latent_dim, dims=(64, 64, 3), fixed_latents=None):
    # put ims through VAE and get predictions
    predictions = get_predictions(input_ims, vae)

    # get high error elements
    diff = get_true_pred_diff(input_ims, predictions)

    # get into right format for MHN
    # Assuming input_ims, diff, and threshold are already defined
    # Compute the average error across the three channels
    avg_diff = np.mean(abs(diff), axis=-1, keepdims=True)
    # Create a mask where the average error is greater than the threshold
    mask = avg_diff > threshold
    # Create the output image using the mask and input images
    version_to_visualise = np.where(mask, (input_ims * 2) - 1, 0)

    if fixed_latents is None:
        _, latents = get_recalled_ims_and_latents(vae, input_ims, noise_level=0)
        latents = np.array([latents[0][i] for i in range(len(latents[0]))])
        latents = latents.reshape((input_ims.shape[0], latent_dim, 1))
    else:
        latents = fixed_latents

    sparse_to_encode = version_to_visualise.reshape((input_ims.shape[0], np.prod(dims), 1))
    hpc_traces = np.concatenate((sparse_to_encode, latents), axis=1)
    return hpc_traces, latents, version_to_visualise


def recall_memories(test_data, net, vae, dims, latent_dim, test_ind=0, noise_factor=0.2, threshold=0.005,
                    return_final_im=False, fixed_latents=None):
    # get noisy input
    noisy_data = noise(test_data[0:10], noise_factor=noise_factor, gaussian=False)

    # get output of MHN, recalled, for this input
    hpc_traces, latents, to_visualise = ims_to_hpc_format(noisy_data, vae, threshold, latent_dim, dims=dims)
    test = hpc_traces[test_ind].reshape(-1, 1)
    recalled = net.retrieve(test, max_iter=5)

    # A handful of images fail with OOM errors - exclude these
    if np.isnan(recalled).all():
        logging.warning("NaN values detected; skipping")
        return None, None

    # extract unpredictable image elements from MHN output
    high_error = recalled[0:np.prod(dims)].reshape((dims[0], dims[1], dims[2]))
    # extract reconstructed latent vector from MHN output
    reconstructed_latent = recalled[np.prod(dims):]

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[3, 1], wspace=0.05, hspace=0.05)

    # Plot original image
    plt.subplot(gs[0])
    plt.imshow(test_data[test_ind])
    plt.axis('off')

    # For Carmichael effect simulations, plot external latent vector
    if fixed_latents is not None:
        plt.subplot(gs[6])
        plt.imshow(fixed_latents.reshape(1, latent_dim), cmap='binary')
        plt.axis('off')

    # Plot noisy image (input to recall)
    plt.subplot(gs[1])
    plt.imshow(noisy_data[test_ind])
    plt.axis('off')

    # Plot unpredictable elements of input
    plt.subplot(gs[2])
    plt.imshow((to_visualise[test_ind] + 1)/2)
    plt.axis('off')

    # Plot latent vector
    plt.subplot(gs[8])
    plt.imshow(latents[test_ind].reshape(1, latent_dim), cmap='binary')
    plt.axis('off')

    # Plot recalled unpredictable elements
    plt.subplot(gs[3])
    plt.imshow((high_error + 1)/2)
    plt.axis('off')

    # Plot recalled latent vector
    plt.subplot(gs[9])
    plt.imshow(reconstructed_latent.reshape(1, latent_dim), cmap='binary')
    plt.axis('off')

    # Plot decoded latent variables
    plt.subplot(gs[4])
    plt.imshow(vae.decoder.predict(reconstructed_latent.reshape(1, latent_dim)).reshape((dims[0], dims[1], dims[2])),
               vmin=0, vmax=1)
    plt.axis('off')

    vae_pred = vae.decoder.predict(reconstructed_latent.reshape(1, latent_dim)).reshape((dims[0], dims[1], dims[2]))

    mask = np.logical_not(np.isclose(high_error, 0, atol=1e-2))
    combined = np.where(mask, high_error, (vae_pred * 2) - 1)

    # Plot final output
    plt.subplot(gs[5])
    plt.imshow((combined + 1)/2)
    plt.axis('off')

    plt.show()

    if return_final_im is True:
        return fig, combined
    else:
        return fig, None


def test_extended_model(dataset='shapes3d', vae=None, beta=6, generative_epochs=100, num_ims=1000, latent_dim=10,
                        threshold=0.01, return_errors_and_counts=False):
    dims = dims_dict[dataset]

    pdf = matplotlib.backends.backend_pdf.PdfPages("./hybrid_model/extended_version_{}_{}.pdf".format(dataset,
                                                                                                      threshold))

    if vae is None:
        net, vae = run_end_to_end(dataset=dataset, generative_epochs=generative_epochs,
                                  num=num_ims, latent_dim=latent_dim, kl_weighting=1)
        logging.info("Trained VAE.")

    train_data, test_data, noisy_train_data, noisy_test_data, train_labels, test_labels = prepare_data(dataset,
                                                                                                       labels=True)
    test_data, test_labels = shuffle(test_data, test_labels, random_state=0)

    # create variant of this dataset with unusual features
    test_data_squares = [add_multiple_white_squares(d, dims, 3) for d in test_data][0:100]
    test_data_squares = np.stack(test_data_squares, axis=0).astype("float32")[0:100]

    # get reconstructed images and predicted labels for test data
    predictions = get_predictions(test_data_squares, vae)
    diff = get_true_pred_diff(test_data_squares, predictions)

    # create MHN for predictable and unpredictable elements
    net = ContinuousHopfield(np.prod(dims) + latent_dim, beta=beta)
    # get elements where error is above threshold and remap to range [-1,1]
    # first compute the average error across the three channels
    avg_diff = np.mean(abs(diff), axis=-1, keepdims=True)
    # next create a mask where the average error is greater than the threshold
    mask = avg_diff > threshold
    # finally create the output image using the mask and input images
    sparse_to_encode = np.where(mask, (test_data_squares * 2) - 1, 0)

    # get latent vectors for test memories
    _, latents = get_recalled_ims_and_latents(vae, test_data_squares, noise_level=0)

    pred_labels = np.array([latents[0][i] for i in range(len(latents[0]))])
    pred_labels = pred_labels.reshape((100, latent_dim, 1))

    # encode traces in MHN
    a = sparse_to_encode.reshape((100, np.prod(dims), 1))

    hpc_traces = np.concatenate((a, pred_labels), axis=1)
    net.learn(hpc_traces[0:50])

    # now visualise recall from a noisy image in the MHN
    images_masked_np = noise(hpc_traces, noise_factor=0.3, gaussian=True)

    # TEMPORARY CODE
    print(images_masked_np.shape)
    images_masked_np = np.random.uniform(-1, 1, size=(100, 12308, 1))

    tests = []
    label_inputs = []
    predictions = []
    pred_labels = []

    for test_ind in range(10):
        test_in = images_masked_np[test_ind].reshape(-1, 1)
        test_out = net.retrieve(test_in, max_iter=5)
        reconstructed = test_out[0:np.prod(dims)]
        input_im = test_in[0:np.prod(dims)]

        predictions.append(np.array(reconstructed).reshape((1, dims[0], dims[1], dims[2])))
        tests.append(np.array(input_im).reshape((1, dims[0], dims[1], dims[2])))
        pred_labels.append(test_out[np.prod(dims):])
        label_inputs.append(test_in[np.prod(dims):])

    predictions = np.concatenate(predictions, axis=0)
    tests = np.concatenate(tests, axis=0)
    pred_labels = np.array(pred_labels).reshape(10, latent_dim)
    label_inputs = np.array(label_inputs).reshape(10, latent_dim)

    fig = display_with_labels(tests, label_inputs, predictions, pred_labels, n_labels=latent_dim)
    pdf.savefig(fig)

    # for the first ten test images, display the stages of recall
    final_outputs = []
    for test_ind in range(10):
        fig, final_im = recall_memories(test_data_squares, net, vae, dims, latent_dim, test_ind=test_ind,
                                        noise_factor=0.1, threshold=threshold, return_final_im=True)

        if fig is not None:
            pdf.savefig(fig)
            final_outputs.append(final_im)

    pdf.close()

    if return_errors_and_counts:
        final_outputs = (np.array(final_outputs) + 1)/2
        errors = np.abs(test_data_squares[0:10] - final_outputs.reshape((10, dims[0], dims[1], dims[2])))
        # Square the errors
        squared_errors = errors ** 2
        # Sum the squared errors for each image
        sum_squared_errors = np.sum(squared_errors, axis=(1, 2, 3))
        # Compute the mean squared error across the 10 images
        mean_error = np.mean(sum_squared_errors)

        pixel_counts = np.where(abs(diff) > threshold, 1, 0)[0:10]
        # Count non-zero elements for each image
        non_zero_counts = np.count_nonzero(pixel_counts, axis=(1, 2, 3))
        # Compute the average count across the 10 images
        mean_pixel_count = np.mean(non_zero_counts)

        return mean_error, mean_pixel_count
