from model import GAN
from utils import check_and_use_gpus, get_mnist_images

if __name__ == "__main__":
    check_and_use_gpus(memory_limit=8192)
    training_data = get_mnist_images()

    gan = GAN(
        training_data=training_data,
        epochs=25,
        batch_size=32,
        optimizer="adam"
    )

    gan.fit()