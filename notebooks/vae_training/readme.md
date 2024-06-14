# Variational Autoencoders for synthetic data generation

# MNIST Variational Autoencoder (VAE) Architecture

## Overview
This project explores the application of Variational Autoencoders (VAEs) on the MNIST dataset. The objective was to utilize custom architecture and theoretical knowledge to effectively train VAEs and observe their performance. The project also leverages Weights & Biases (wandb) for monitoring the training process and PyTorch for model training.

## Installation and Setup
To get started with the project, ensure that you have the necessary libraries installed. The primary libraries used in this project are:

- PyTorch for building and training the neural network.

## Dataset Preparation

The MNIST dataset, a well-known benchmark for image processing tasks, is used in this project. The dataset consists of 60,000 training images and 10,000 testing images of handwritten digits, each of size 28x28 pixels. The images are normalized and transformed into tensors for input into the VAE model.

## Model Architecture

The custom VAE architecture designed for this project includes:

- An encoder network that compresses the input images into a lower-dimensional latent space.
- A decoder network that reconstructs the images from the latent space representation.
- Reparameterization trick to sample from the latent space.
The architecture ensures that the encoded latent space follows a normal distribution, which is essential for generating new, meaningful data points.

## Training Process

- Defining the loss function, which combines reconstruction loss and KL-divergence to ensure the latent space follows the desired distribution.
- Implementing the training loop to optimize the model parameters using backpropagation.

# VAE for Synthetic Data

## Overview
In this notebook, a Variational Autoencoder (VAE) model is trained to generate augmented versions of original images. The aim is to utilize the VAE for creating synthetic data, which can be useful for various data augmentation purposes.

## Installation and Setup
Similar to the previous notebook, ensure that the following libraries are installed:

- PyTorch for model training.
- Weights & Biases (wandb) for experiment tracking.
  
## Dataset Preparation
- The dataset used in this project includes images that require augmentation. 
- The preprocessing steps involve normalizing the images and transforming them into tensors suitable for input into the VAE model.

## Model Architecture 
- An encoder network to map input images to a latent space.
- A decoder network to reconstruct images from the latent space.
- The reparameterization trick for sampling from the latent space.

## Training Process 

- Defining the combined loss function for reconstruction and KL-divergence.
- Implementing a training loop with optimization through backpropagation.
- Using wandb to monitor the training process, including loss values and generated images.


## Results and Visualization
The generated synthetic images are logged and visualized using wandb. This provides insights into the quality of the augmented data and helps in refining the model for better performance.