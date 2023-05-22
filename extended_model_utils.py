from utils import noise
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
import hashlib


def get_recalled_ims_and_latents(vae, test_data, noise_level=0):
    test_data = noise(test_data, noise_factor=noise_level)
    latents = vae.encoder.predict(test_data)
    predictions = vae.decoder.predict(latents[0])
    return predictions, latents


def latent_variable_to_label(latents, labels):
    clf = make_pipeline(StandardScaler(), SVC())
    clf.fit([latents[0][i] for i in range(len(latents[0]))], labels)
    return clf


def deterministic_seed(image):
    hash_object = hashlib.sha1(image.tobytes())
    return int(hash_object.hexdigest(), 16) % (10 ** 8)


def add_white_square(d, dims, seed=0):
    random.seed(deterministic_seed(d))
    square_size = int(dims[0]/8)
    im1 = Image.fromarray((d * 255).astype("uint8"))
    im2 = Image.fromarray((np.ones((square_size, square_size, 3)) * 255).astype("uint8"))
    Image.Image.paste(im1, im2, (random.randrange(0, dims[0]), random.randrange(0, dims[0])))
    return (np.array(im1) / 255).reshape((dims[0], dims[0], 3))


def blend_images(src, dst, alpha):
    return (src * alpha) + (dst * (1 - alpha))


def add_multiple_white_squares(d, dims, n, seed=4321):
    random.seed(seed)
    square_size = int(dims[0] / 8)
    im1 = (d * 255).astype("uint8")

    for _ in range(n):
        # Create a new white square with random transparency
        transparency = random.randint(0, 255) / 255
        im2 = np.ones((square_size, square_size, 3)) * 255

        x = random.randrange(0, dims[0] - square_size)
        y = random.randrange(0, dims[0] - square_size)

        # Blend the white square with the original image using alpha compositing
        im1[y:y+square_size, x:x+square_size] = blend_images(im2, im1[y:y+square_size, x:x+square_size], transparency).astype("uint8")

    return (im1 / 255).reshape((dims[0], dims[0], 3))


def get_predictions_and_labels(input_data, vae, clf):
    latents = vae.encoder.predict(input_data)
    predictions = vae.decoder.predict(latents[0])
    labels = clf.predict([latents[0][i] for i in range(len(latents[0]))])
    return predictions, labels


def get_true_pred_diff(input_data, predictions):
    diff = input_data - predictions
    return diff


def display_with_labels(array1, array1_labels, array2, array2_labels, seed=None,
                        title='Inputs and outputs of the model', random_seed=0, n=10,
                        n_labels=10):

    dim = array1[0].shape[0]
    # Displays ten random images from each one of the supplied arrays.
    if seed is not None:
        np.random.seed(seed)

    np.random.seed(random_seed)
    indices = range(0, n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    labels1 = array1_labels[indices, :]
    labels2 = array2_labels[indices, :]

    fig = plt.figure(figsize=(20, 4))
    for i, (image1, image2, label1, label2) in enumerate(zip(images1, images2, labels1, labels2)):
        ax = plt.subplot(4, n, i + 1)
        plt.imshow((image1+1)/2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(label1.reshape(1, n_labels), cmap='binary', vmin=-1, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow((image2+1)/2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(label2.reshape(1, n_labels), cmap='binary', vmin=-1, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle(title)
    plt.show()
    return fig