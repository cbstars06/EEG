import os
from random import randint

import PIL
import numpy as np
from PIL import Image

IMAGE_CLASSES = {'Apple': 0, 'Car': 1, 'Dog': 2, 'Gold': 3, 'Mobile': 4, 'Rose': 5, "Scooter": 6, 'Tiger': 7, 'Wallet': 8, 'Watch': 9}

def randomize(samples, labels):
    if type(samples) is np.ndarray:
        permutation = np.random.permutation(samples.shape[0])
        shuffle_samples = samples[permutation]
        shuffle_lables = labels[permutation]
    else:
        permutation = np.random.permutation(len(samples))
        shuffle_samples = [samples[i] for i in permutation]
        shuffle_lables = [labels[i] for i in permutation]

    return (shuffle_samples, shuffle_lables)

def to_categorical(y, num_classes=None, dtype="float32"):

    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.
    Example:
    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]
    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def load_image_data(imagenet_folder, patch_size):

    images = []
    labels = []
    
    for image_folder in os.listdir(imagenet_folder):
        image_class = IMAGE_CLASSES[image_folder]
        for image_file in os.listdir(os.path.join(imagenet_folder, image_folder)):
            file_path = os.path.join(imagenet_folder, image_folder, image_file)
            img = Image.open(file_path).convert('RGB').resize(patch_size, PIL.Image.ANTIALIAS)
            img_array = np.array(img)
            img_array = img_array/255.0
            images.append(img_array)
            labels.append(image_class)
            images.append(np.flip(img_array, 1))
            labels.append(image_class)
    
    print(len(images), len(labels))
    images, labels = randomize(images, labels)

    images = np.array(images)
    labels = np.array(labels)
    train_size = int(3 * len(images)/4)
    x_train, y_train = images[0: train_size], labels[0: train_size]
    x_test, y_test = images[train_size:], labels[train_size:]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test