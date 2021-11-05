# Project Description
The GAN (Generative Adversarial Netwrok) algorithm is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in June 2014. It's based on "Game Theory", to make two neural networks contest with each other.

In this project, it will restore image using GAN model, and here is how it works:
1. Model setup:
    - Build discriminator and generator by using high-level API of Tensorflow 2
2. Train model:
    1. Yields batches of images from `training_data`. The `training_data`'s shape is `(image_count, image_width, image_hight, image_channel)`
    2. Customize the loss function of discriminator and generator
        - discriminator's loss function
            - $L_D \triangleq -[\log D(X) + \log(1 - D(G(Y)))]$
        - generator's loss function
            - $L_G \triangleq \| G(Y) - X \|_1 + \frac{1}{1000} \log(1 - D(G(Y)))$
    3. Gradient descent with respect to variables of discriminator and generator
        - using `tensorflow.GradientTape` to implement gradient descent
    4. Plot training progress bar in terminal
        -  using `rich` packages of Python to plot `epochs`, `completeness`, `generator loss` and `discriminator loss`
    5. Save model structure and parameters when it finish model training
3. Image Restoration
    1. Load trained model
    2. Get any image fits `training_data`'s shape, e.g. `(image_count, image_width, image_hight, image_channel)`
    3. Restore image

# Project Setup
To avoid TensorFlow version conflicts, the project use pipenv (Python vitural environment) to install Python packages.

> **Notice**: Before executing the following command, please refer to this [website](https://www.tensorflow.org/install/source#linux) and modify the TensorFlow version in `Pipfile`

```
pip install pipenv
pipenv shell
pipenv install
```

# Train Model
In model training stage, you can modify the hyper parameter like epochs, learning_rate, learning_rate_decay, etc.

```
python src/train.py
```

# Predict
You can use the following model to restore images:
- the example model `generator_example.h5` at `src/model/trained_model/` 
- [Other trained model on Google Drive (still building...)](https://drive.google.com/drive/folders/1d431KDCVXYkCfmrGskXQ5vD4FXIJ8nUH?usp=sharing)

```
python src/predict.py
```