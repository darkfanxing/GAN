# Disable info log of TensorFlow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import Tensor, GradientTape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Add, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras import backend as K
from numpy import ndarray, random
from math import ceil
from utils import build_training_progress_bar, generate_batch_random_images_with_mask
from typing import Tuple

class GAN():
    """
    A Generative Adversarial Network (GAN) Model for image restoration.

    This algorithm looks only at the features (input), not the desired
    outputs (label), and can thus be used for unsupervised learning.

    Parameters
    ----------
    training_data: np.ndarray
        the input of model,it should be the same shape as:
            (image_count, image_width, image_height, image_channel)
    
    epochs: int
        An epoch means training the neural network with all the training
        data for one cycle.
    
    batch_size: int
        A number of samples processed before the model is updated.
    
    optimizer: str, default="adam"
        A algorithm used to change the attributes of model such as
        weights and learning rate in order to reduce the losses. It
        is supported "adam" Algorithm now.

    generator_learning_rate: float default=1e-4
        The learning_rate of generator's optimizor

    discriminator_learning_rate: float, default=1e-4
        The learning_rate of discriminator's optimizor

    learning_rate_decay: flaot, default=1e-8
        It will slowly decay optimizor's learning rate.

    Methods
    -------
    fit()
        Train the GAN model from the training data, and then save the
        model structure and parameter to:
            - `src/model/trained_model/gan_discriminator/`
            - `src/model/trained_model/gan_generator/`

    predict(images)
        The generator will restorate the images with mask. the images'
        shape should be:
            (image_count, image_width, image_height, image_channel)

    Examples
    --------
        >>> from utility import get_mnist_images
        >>> training_data = get_mnist_images()
        >>> model = GAN(
        ...     training_data=training_data,
        ...     epochs=20,
        ...     batch_size=32,
        ...     optimizer="adam"
        ... )
        >>> model.fit()
        epoch 1/20  ━━                        66/1875 - ETA:  0:03:51 - Generator Loss: 642116.0293 - Discriminator Loss: 376.3615
        >>> restorated_images = model.predict(training_data[:10]) # prediect 10 images
    """
    def __init__(
        self,
        training_data: ndarray,
        epochs: int,
        batch_size: int,
        optimizer: str = "adam",
        generator_learning_rate: float = 1e-4,
        discriminator_learning_rate: float = 1e-4,
        learning_rate_decay: float = 1e-8
    ):
        self.training_data = training_data
        self.epochs = epochs
        self.batch_size = batch_size
        
        if optimizer == "adam":
            self._generator_optimizor: Optimizer = Adam(
                learning_rate=generator_learning_rate,
                decay=learning_rate_decay
            )
            self._discriminator_optimizor: Optimizer = Adam(
                learning_rate=discriminator_learning_rate,
                decay=learning_rate_decay
            )

        self._build_discriminator()
        self._build_generator()


    def _build_discriminator(self) -> None:
        input_x: Tensor = Input(shape=self.training_data.shape[1:])
        x = Conv2D(filters=256, kernel_size=(4, 4))(input_x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=256, kernel_size=(4, 4))(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=256, kernel_size=(4, 4))(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=512, kernel_size=(4, 4))(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=1, kernel_size=(4, 4))(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=1, kernel_size=(6, 6))(x)
        x = Flatten()(x)
        y = Dense(units=1, activation="sigmoid")(x)
        
        self._discriminator: Model = Model(inputs=input_x, outputs=y, name="Discriminator")


    def _build_generator(self) -> None:
        def residual_block(input_x: Tensor) -> Tensor:
            x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(input_x)
            x = LeakyReLU()(x)
            x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), padding="same")(x)
            x = Add()([x, input_x])
            x = LeakyReLU()(x)
            return x

        image_dimension = self.training_data.shape[3]

        input_x: Tensor = Input(shape=self.training_data.shape[1:])
        x = Conv2D(filters=image_dimension, kernel_size=(7, 7), padding="same")(input_x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2))(x)
        x = LeakyReLU()(x)
        x = residual_block(x)
        x = residual_block(x)
        x = residual_block(x)
        x = residual_block(x)
        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), output_padding=(1, 1))(x)
        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2))(x)
        x = Dropout(0.3)(x)
        y = Conv2D(filters=image_dimension, kernel_size=(7, 7), padding="same")(x)

        self._generator: Model = Model(inputs=input_x, outputs=y, name="Generator")


    def _discriminator_loss_function(self, validity_of_real_image: Tensor, validity_of_fake_image: Tensor) -> Tensor:
        loss_of_real_image = K.log(validity_of_real_image + K.epsilon())
        loss_of_fake_image = K.log(1 - validity_of_fake_image + K.epsilon())
        return -K.sum(loss_of_real_image + loss_of_fake_image)


    def _generator_loss_function(self, real_images: ndarray, fake_images: ndarray, validity_of_fake_image: Tensor) -> Tensor:
        print(type(fake_images))
        images_similarity = K.sum(K.abs(fake_images - real_images))
        images_fidlity = K.sum(0.001*K.log(1 - validity_of_fake_image + K.epsilon()))
        return images_similarity + images_fidlity


    def _fit(self, real_images: ndarray) -> Tuple[Tensor, Tensor]:
        with GradientTape() as generator_gradient_tape, GradientTape() as discriminator_gradient_tape:
            validity_of_real_images = self._discriminator(real_images, training=True)

            images_with_mask = generate_batch_random_images_with_mask(real_images)
            fake_images = self._generator(images_with_mask, training=True)
            vailidty_of_fake_images = self._discriminator(fake_images, training=True)

            discriminator_loss = self._discriminator_loss_function(validity_of_real_images, validity_of_real_images)
            generator_loss = self._generator_loss_function(real_images, fake_images, vailidty_of_fake_images)

            discriminator_loss_gradient = discriminator_gradient_tape.gradient(
                target=discriminator_loss,
                sources=self._discriminator.trainable_variables
            )
            generator_loss_gradient = generator_gradient_tape.gradient(
                target=generator_loss,
                sources=self._generator.trainable_variables
            )

            self._discriminator_optimizor.apply_gradients(
                zip(discriminator_loss_gradient, self._discriminator.trainable_variables)
            )
            self._generator_optimizor.apply_gradients(
                zip(generator_loss_gradient, self._generator.trainable_variables)
            )

            return generator_loss, discriminator_loss


    def fit(self) -> None:
        progress_bar = build_training_progress_bar()
        for epoch in range(self.epochs):
            batch_count = ceil(self.training_data.shape[0]/self.batch_size)
            
            task = progress_bar.add_task(
                description=f"epoch {epoch}/{self.epochs}",
                total=batch_count,
                generator_loss=0,
                discriminator_loss=0,
                batch_index=0
            )
            with progress_bar:
                images = self.training_data.copy()
                random.shuffle(images)
                for batch_index in range(batch_count):
                    start_data_index = self.batch_size*batch_index
                    end_data_index = self.batch_size * (batch_index+1)
                    generator_loss, discriminator_loss = self._fit(real_images=images[start_data_index:end_data_index])
                    
                    progress_bar.update(
                        task,
                        advance=1,
                        generator_loss=generator_loss,
                        discriminator_loss=discriminator_loss,
                        batch_index=batch_index
                    )

            self._discriminator.save(f"src/model/trained_model/gan_discriminator/{epoch}epochs.h5")
            self._generator.save(f"src/model/trained_model/gan_generator/{epoch}epochs.h5")


    def predict(self, images: ndarray) -> ndarray:
        return self._generator.predict(images)