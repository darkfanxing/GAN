## Table of contents
- [Project Description](#project-description)
- [Project Setup](#project-setup)
- [How To Train Model In this Project](#how-to-train-model-in-this-project)
- [How To Restore Images In this Project](#how-to-restore-images-in-this-project)

## Project Description
The GAN (Generative Adversarial Netwrok) algorithm is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in June 2014. It's based on "Game Theory", to make two neural networks contest with each other.

In this project, it will restore image using GAN model, and here is how it works:
1. Model setup:
    - Build discriminator and generator by using high-level API of Tensorflow 2
2. Train model:
    1. Yields batches of images from `training_data`. The `training_data`'s shape is `(image_count, image_width, image_hight, image_channel)`
    2. Put the random mask over data (each picture)
    3. Customize the loss function of discriminator and generator
        - discriminator's loss function
            - ![](https://i.imgur.com/bd0OoXI.png)
        - generator's loss function
            - ![](https://i.imgur.com/TbQ7Fia.png)
    4. Gradient descent with respect to variables of discriminator and generator
        - using `tensorflow.GradientTape` to implement gradient descent
    5. Plot training progress bar in terminal
        -  using `rich` packages of Python to plot `epochs`, `completeness`, `generator loss` and `discriminator loss`
    6. Save model structure and parameters when it finish model training
3. Image Restoration
    1. Load trained model
    2. Get any image with mask fits `training_data`'s shape, e.g. `(image_count, image_width, image_hight, image_channel)`
    3. Restore image

## Project Setup
To avoid TensorFlow version conflicts, the project use pipenv (Python vitural environment) to install Python packages.

> **Notice**: Before executing the following command, please refer to this [website](https://www.tensorflow.org/install/source#linux) and modify the TensorFlow version in `Pipfile`

```console
pip install pipenv
pipenv shell
pipenv install
```

## How To Train Model In This Project
In model training stage, you can modify the hyper parameter like epochs, learning_rate, learning_rate_decay, etc.

```console
python src/train.py
```

## How To Restore Images In This Project
You can use the following model to restore images:
- The example model `generator_example.h5` at `src/model/trained_model/` 
- [Other trained model on Google Drive (still building...)](https://drive.google.com/drive/folders/1d431KDCVXYkCfmrGskXQ5vD4FXIJ8nUH?usp=sharing)

```console
python src/predict.py
```
