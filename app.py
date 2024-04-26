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
    #############getting data from ui
    data = request.get_json() 
    lines=data['lines']
    resolution=data['resolution']
    image_data = data['image_data']
    #########################################
    # Split the base64 image data from the data URL
     # Decode and process image data
    image_data = base64.b64decode(image_data.split(',')[1])
    # Convert image data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))
########################################################################################
########################################################################################
##setting up the image
    original_image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)

    greyscale_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2GRAY)
    
    ###############################################################################
    ################       HOUGH LINE DETECTION     ################################
    lines = functions.HoughLine( greyscale_image, lines, resolution)
    output_image=functions.transformToImageSpace(greyscale_image, lines)
    ####################################################################################
    output_path = os.path.join(os.path.dirname(__file__), 'image.png')
    cv2.imwrite(output_path, output_image)

    # Return the processed image file
    return send_file('image.png', mimetype='image/png')

    
@app.route('/upload2', methods=['POST'])
def upload2():
        #############getting data from ui

    data = request.get_json() 
    threshold=data['Threshold'] 
    image_data = data['image_data']
    image_data = base64.b64decode(image_data.split(',')[1])
    # Convert image data to a PIL Image object
    # Split the base64 image data from the data URL
     # Decode and process image data
    image = Image.open(io.BytesIO(image_data))

    original_image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)

    greyscale_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2GRAY)

    ###############################################################################
    #################  HARRIS CORNER DETECTION  ####################################
    output_image = functions.harris_corner_detection(original_image = original_image_rgb, greyscale_image = greyscale_image, threshold= threshold)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    ####################################################################################
    output_path = os.path.join(os.path.dirname(__file__), 'image2.png')
    cv2.imwrite(output_path, output_image)

    # Return the processed image file
    return send_file('image2.png', mimetype='image/png')

    

    


if __name__ == '__main__':
    app.run(debug=True)