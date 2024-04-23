from flask import Flask, render_template,request, jsonify, send_file
import base64
from PIL import Image
import io
import cv2
import numpy as np
import os
import functions  # Import custom functions from cannyfuncyions module


app = Flask(__name__)

@app.route('/')
def CannyEdgeDetector():
    return render_template('main.html')


@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json() 
    lines=data['lines']
    resolution=data['resolution']
    image_data = data['image_data']
    
     # Decode and process image data
    image_data = base64.b64decode(image_data.split(',')[1])
    # Convert image data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))

    # Convert RGB image to BGR format
    cv2_image = functions.convert_rgb_to_bgr(image)

    # Convert BGR image to grayscale
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    ###############################################################################
    
    
    ####################################################################################
    output_path = os.path.join(os.path.dirname(__file__), 'image.png')
    cv2.imwrite(output_path, cv2_image)

    # Return the processed image file
    return send_file('image.png', mimetype='image/png')

    
@app.route('/upload2', methods=['POST'])
def upload2():
    data = request.get_json() 
    threshold=data['Threshold'] 
    image_data = data['image_data']
    image_data = base64.b64decode(image_data.split(',')[1])
    # Convert image data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))

    # Convert RGB image to BGR format
    cv2_image = functions.convert_rgb_to_bgr(image)

    # Convert BGR image to grayscale
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    ###############################################################################
    
    
    ####################################################################################
    output_path = os.path.join(os.path.dirname(__file__), 'image2.png')
    cv2.imwrite(output_path, cv2_image)

    # Return the processed image file
    return send_file('image2.png', mimetype='image/png')

    

    


if __name__ == '__main__':
    app.run(debug=True)