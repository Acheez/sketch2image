# Real Estate Image Processing 

##  Overview
This folder consists of three main components:

- **Scraping Redfin.com**: Scrape property listing links from Redfin.com and download associated images.
- **Filtering Images**: Use a pre-trained YOLO model to filter out images that do not contain desired objects.
- **Image to Sketch Conversion**: Convert property images to sketch-like images using Sobel edge detection.

## Prerequisites

- Python 3.x
- requests
- BeautifulSoup
- concurrent.futures
- os
- hashlib
- cv2 (OpenCV)
- tqdm
- ultralytics
- numpy

```cmd
pip install requests beautifulsoup4 tqdm ultralytics opencv-python-headless numpy
```

## Notebooks

### Filtering Images (data_filter.ipynb)

**Purpose**: Filter out images that do not contain desired objects using a pre-trained YOLO model.

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. It can detect multiple objects in an image and classify them in a single forward pass of the network. YOLO is known for its speed and accuracy, making it ideal for applications requiring real-time object detection.



Steps:

1. Load YOLO Model:

   - Load a pre-trained YOLO model, which is trained to detect specific objects.
   - Example: `model = YOLO("best.pt")` where `best.pt` is the pre-trained model file.
2. Process Images:
   - Iterate through the downloaded images.
   - Use the YOLO model to detect objects in each image.
   - If no objects are detected in an image, the image is removed.
   - Example:
     ```python
     results = model.predict(image)
     if len(results[0].boxes) == 0:
         os.remove(image_path)
     ```

### Image to Sketch Conversion (im2sketch.ipynb)

**Purpose**: Convert property images to sketch-like images using Sobel edge detection.
**Functions:**
- **Grayscale Conversion**: Converts a color image to grayscale.
    ```python
    def grayscale(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img
    ```
- **Gaussian Blur:** Applies a Gaussian blur to a grayscale image to reduce noise and detail.
    ```python
    def blur(gray_img, blur_ksize):
        blurred_img = cv2.GaussianBlur(gray_img, (blur_ksize, blur_ksize), 0)
        return blurred_img
    ```
- **Resize Image:** Resizes the image by a given scale percentage.
    ```python
    def resize_image(img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized_img
    ```

- **Sobel Edge Detection:** Detects edges in the blurred grayscale image using the Sobel operator.
    ```python
    def sobel_edges(blurred_img):
        grad_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        return grad
    ```

## Usage

1. Run *data_filter.ipynb*:
   - Load the pre-trained YOLO model.
   - Process the downloaded images to filter out unwanted ones.
2. Run *im2sketch.ipynb*:
   - Use the provided functions to convert images to sketches.
   - Display the sketches for verification.
