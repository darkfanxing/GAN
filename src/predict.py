# Disable info log of TensorFlow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils import check_and_use_gpus, generate_batch_random_images_with_mask, get_mnist_images
from tensorflow.keras.models import load_model
from cv2 import imshow, waitKey
import random

if __name__ == "__main__":
    check_and_use_gpus(memory_limit=8192)
    test_data = get_mnist_images()

    model = load_model("src/model/trained_model/generator_example.h5")

    for index in random.sample(
        population=range(0, test_data.shape[0]),
        k=10
    ):
        test_data_with_mask = generate_batch_random_images_with_mask(test_data[index].reshape(1, 28, 28, 1))
        generated_image = model.predict(test_data_with_mask)

        imshow("original image", test_data[index])
        imshow("original image with mask", test_data_with_mask.reshape((28, 28)))
        imshow("generated image", generated_image.reshape((28, 28)))
        waitKey(0)