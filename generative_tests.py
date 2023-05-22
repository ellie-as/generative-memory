import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from utils import display, noise, prepare_data


def check_generative_recall(vae, test_data, noise_level=0.1):
    test_data = noise(test_data, noise_factor=noise_level)
    latents = vae.encoder.predict(test_data)
    predictions = vae.decoder.predict(latents[0])
    fig = display(test_data, predictions, title='Inputs and outputs for VAE')
    return fig


def plot_history(history, decoding_history, titles=False):
    recon_loss_values = history.history['reconstruction_loss']
    decoding_acc_values = decoding_history.decoding_history
    epochs = range(1, len(recon_loss_values)+1)

    fig, ax = plt.subplots(figsize=(3,3))
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax.set_ylabel("Reconstruction error")
    ax2.set_ylabel("Decoding accuracy")

    ax.plot(epochs, recon_loss_values, label='Reconstruction Error', color='red')
    ax2.plot(epochs, decoding_acc_values, label='Decoding Accuracy', color='blue')

    if titles is True:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        plt.title('Reconstruction error and decoding accuracy over time')

    ax.set_xlabel('Epoch')
    plt.show()
    return fig


def interpolate_ims(latents, vae, first, second):
    encoded_imgs = latents[0]
    enc1 = encoded_imgs[first:first+1]
    enc2 = encoded_imgs[second:second+1]

    linfit = interp1d([1, 10], np.vstack([enc1, enc2]), axis=0)

    fig = plt.figure(figsize=(20, 5))

    for j in range(10):
        ax = plt.subplot(1, 10, j+1)
        decoded_imgs = vae.decoder.predict(np.array(linfit(j+1)).reshape(1,encoded_imgs.shape[1]))
        ax.imshow(decoded_imgs[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle('Interpolation between items')
    plt.show()
    return fig


def vector_arithmetic(imgs, latents, vae, first, second, third):
    img1 = imgs[first]
    img2 = imgs[second]
    img3 = imgs[third]

    encoded_imgs = latents[0]
    enc1 = encoded_imgs[first:first+1]
    enc2 = encoded_imgs[second:second+1]
    enc3 = encoded_imgs[third:third+1]

    fig, axs = plt.subplots(1, 4, figsize=(10,2))
    axs[0].imshow(img1)
    axs[0].axis('off')
    axs[1].imshow(img2)
    axs[1].axis('off')
    axs[2].imshow(img3)
    axs[2].axis('off')
    # enc1-enc2=enc3-enc4 -> enc4=enc3+enc2-enc1
    res = - enc1 + enc2 + enc3
    axs[3].imshow(vae.decoder.predict([res])[0])
    axs[3].axis('off')
    fig.suptitle('Vector arithmetic')
    plt.show()
    return fig


def plot_latent_space_with_labels(latents, labels, titles=False):
    np.random.seed(1)
    fig = plt.figure(figsize=(4, 4))

    embedded = TSNE(n_components=2, init='pca').fit_transform(latents[0][0:800])
    x = [x[0] for x in embedded]
    y = [x[1] for x in embedded]

    plt.scatter(x, y, c=labels[0:800], alpha=0.5, cmap=plt.cm.plasma)
    if titles is True:
        plt.title('Latent space in 2D, colour-coded by label')
    plt.show()
    return fig

