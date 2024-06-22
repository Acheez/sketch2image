from config import IMG_HEIGHT, IMG_WIDTH
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import load_img, array_to_img
import matplotlib.pyplot as plt 
import cv2
def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image,channels=3)
  image = tf.cast(image, tf.float32)
  return image

def load_image(image_file):
  input_image = load(image_file)
  input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
  input_image = normalize(input_image)

  return input_image

def display_tensorflow_image(img):
  to_img = array_to_img(img)
  to_img.show()

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
    
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def generate_images(model, test_input):
  prediction = model(test_input, training=True)
  return prediction[0]


def grayscale(img):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return gray_img

def blur(gray_img, blur_ksize):
    blurred_img = cv2.GaussianBlur(gray_img, (blur_ksize, blur_ksize), 0)
    return blurred_img

def resize_image(img, width, height):
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

def sobel_edges(blurred_img):
    grad_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)

    grad = cv2.magnitude(grad_x, grad_y)

    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    return grad

def generate_sketches(image_path, blur_ksize=5):
    img = cv2.imread(image_path)
    img = resize_image(img, IMG_WIDTH, IMG_HEIGHT)  # Resize image to 256x256

    gray_img = grayscale(img)
    blurred_img = blur(gray_img, blur_ksize)
    edges = sobel_edges(blurred_img)
    sketch = 255 - edges  

    return sketch

