from os import getcwd

from mnist import MNIST

from numpy import array

from tqdm import tqdm


def get_mnist_data():
    """
    Open mnist files and load the data in form of lists.
    Combine the training and testing into one.
    """
    data = MNIST(getcwd() + '/data')
    train_images, train_labels = data.load_training()
    test_images, test_labels = data.load_testing()
    images = train_images + test_images
    labels = list(train_labels) + list(test_labels)
    return images, labels

images, labels = get_mnist_data()


def one_hot_encode(integer, array_length):
    """
    One-Hot-Encode a digit.
    For example, 0 becomes [1,0,0,0,0,0,0,0,0,0]
    """
    base = [0 for i in range(array_length)]
    base[integer] = 1
    return base


print('\nOne-Hot-Encode Outputs\n')
labels = array([one_hot_encode(label, 10) for label in tqdm(labels)])


def normalize_and_reshape(image):
    return array([float(i) / 255 for i in image]).reshape(28, 28)


print('\nNormalize and reshape input images\n')
images = array([normalize_and_reshape(image) for image in tqdm(images)])


def split_data():
    """
    Split data into training and testing.
    Training will have 40000 datapoints.
    Testing will have 30000 datapoints.
    Reshape the arrays as the network would expect.
    """
    x_train, y_train = images[:40000].reshape(
        40000, 28, 28, 1), labels[:40000]
    x_test, y_test = images[40000:].reshape(
        30000, 28, 28, 1), labels[40000:]
    return x_test, x_train, y_test, y_train

x_test, x_train, y_test, y_train = split_data()
