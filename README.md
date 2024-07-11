# Canny Edge and Harris Corner Detection Web Application

This project is a web application built with Flask that performs Canny edge detection and Harris corner detection on uploaded images. Users can upload an image and specify parameters for the detection algorithms, and the processed image will be returned.

## Project Structure

- `app.py`: Main application file containing the Flask server and route definitions.
- `functions.py`: Contains custom functions for image processing including Canny edge detection and Harris corner detection.
- `templates/`: Directory containing HTML templates.
- `static/`: Directory containing JavaScript.

## Installation

1. Clone the repository

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Ensure you are in the project directory and the virtual environment is activated.
2. Run the Flask application:
    ```bash
    python app.py
    ```

3. Open your browser and go to `http://127.0.0.1:5000/`.

## Usage

### Routes

- `/`: Renders the main HTML page where users can upload images and set parameters.
- `/upload`: Handles image uploads for Canny edge detection and returns the processed image.
- `/upload2`: Handles image uploads for Harris corner detection and returns the processed image.

### Parameters

For Canny edge detection:
- `lines`: Number of lines to detect.
- `resolution`: Resolution for the Hough transform.
- `image_data`: Base64 encoded image data.

For Harris corner detection:
- `Threshold`: Threshold value for corner detection.
- `image_data`: Base64 encoded image data.

## Functions

### `functions.py`

#### `convert_rgb_to_bgr(image)`
Converts an RGB image to BGR format.

#### `harris_corner_detection(original_image, greyscale_image, window_size=9, k=0.04, threshold=0.5)`
Performs Harris corner detection on the input image.

#### `HoughLine(img, numberOfLines, resolution)`
Performs Hough line detection on the input image.

#### `transformToImageSpace(img, lines)`
Transforms detected lines back to image space and overlays them on the original image.

## Dependencies

- Flask
- OpenCV
- NumPy
- PIL (Pillow)
- SciPy

## Authors

| Name | GitHub | LinkedIn |
| ---- | ------ | -------- |
| Omar Adel Hassan | [@Omar_Adel](https://github.com/omar-adel1) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omar-adel-59b707231/) |
| Sharif Ehab | [@Sharif_Ehab](https://github.com/SharifEhab) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sharif-elmasry-b167a3252/) |
| Mostafa Khaled | [@Mostafa_Khaled](https://github.com/MostafaDarwish93) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mostafa-darwish-75a29225b/) |
| Bahey Ismail | [@Bahey_Ismail ](https://github.com/Bahey1200022) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bahey-ismail-1602431a4/) |
