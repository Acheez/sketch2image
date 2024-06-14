# Sketch2Image Model Training 

**Purpose**: Train a Pix2Pix model for sketch-to-image conversion using TensorFlow.

- *What is Pix2Pix?* - Pix2Pix is a type of conditional Generative Adversarial Network (cGAN) designed for image-to-image translation tasks. It consists of a generator and a discriminator network. The generator creates images based on input sketches, while the discriminator evaluates the realism of the generated images compared to real images.
  
- W*hy Pix2Pix for Sketch-to-Image Conversion?* - Pix2Pix is effective for sketch-to-image conversion because it can learn the mapping between input sketches and output images, preserving important features and textures.
  
**Steps:**
1. Dataset Preparation:
    - The dataset is prepared by concatenating the source (sketch) and target (real image) side by side.
    - Example:
    ```python
    def load(image_file):
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)
        return input_image, real_image
    ```
2. Model Training:
    - Define the generator and discriminator models.
    - Implement loss functions for both models 
    - Train the models using the prepared dataset
3. Training Loop
   - Run the training loop to fit model on the dataset 
   - Save checkpoints and display intermediate results