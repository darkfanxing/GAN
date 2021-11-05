from numpy import ndarray, expand_dims
from tensorflow.keras.datasets import mnist

def get_mnist_images() -> ndarray:
    (images, _), (_, _) = mnist.load_data()
    images = expand_dims(images, axis=3)
    return images