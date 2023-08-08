import numpy as np
from PIL import Image

from config import dims_dict
from hopfield_models import ContinuousHopfield, DenseHopfield, ClassicalHopfield
from utils import load_tfds_dataset


def create_hopfield(num, hopfield_type='continuous', dataset='mnist', beta=10):
    images = load_images_dataset(num, dataset=dataset)
    images_np = convert_images(images)
    images_np = [im_np.reshape(-1, 1) for im_np in images_np]

    n_pixel = dims_dict[dataset][0]
    n_channels = 3
    orig_shape = n_pixel, n_pixel, n_channels
    n = np.prod(orig_shape)
    train_patterns = images_np

    if hopfield_type == 'continuous':
        net = ContinuousHopfield(n, beta=beta)
    if hopfield_type == 'dense':
        net = DenseHopfield(n, beta=beta)
    if hopfield_type == 'classical':
        net = ClassicalHopfield(n)

    net.learn(train_patterns)
    return net


def load_images_dataset(num, dataset='mnist'):
    train_data, test_data = load_tfds_dataset(dataset)
    np.random.shuffle(train_data)
    images = []
    for i in range(num):
        im_arr = train_data[i]
        im = Image.fromarray(im_arr)
        images.append(im)
    return images


def convert_images(images):
    # converts images with values 0 to 255 to ones with values -1 to 1
    images_np = []
    for im in images:
        im_np = ((np.asarray(im) / 255) * 2) - 1
        images_np.append(im_np)
    return images_np


def mask_image_random(n):
    random_arrays = []
    for i in range(n):
        random_array = np.random.uniform(-1, 1, size=(64, 64, 3))
        random_arrays.append(random_array)
    return random_arrays
