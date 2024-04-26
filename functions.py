import numpy as np
import scipy.stats as st
import cv2
import scipy.ndimage as ndimage

# Function to convert an RGB image to BGR format
def convert_rgb_to_bgr(image):
    """
    Convert an RGB image to BGR format using OpenCV's cv2.cvtColor() function.
    
    Parameters:
    - image: NumPy array representing the input RGB image.
    
    Returns:
    - bgr_image: NumPy array representing the image converted to BGR format.
    """
    bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return bgr_image
    

# 


# Perform Harris Corner Detection from Scratch

def harris_corner_detection(original_image,greyscale_image, window_size=9, k=0.04, threshold=0.5):
    """
    Perform Harris corner detection on the input image.

    Args:
    - image (ndarray): Input image.
    - window_size (int): Size of the window for calculating the corner response function.
    - k (float): Empirical constant to be used in the Harris detector; k should lie between 0.04 and 0.06 for good results.
    - threshold (float): Threshold for corner response function.

    Returns:
    - corners (ndarray): Image with detected corners.
    The result of Harris Corner Detection is a grayscale image with this score as the intensity of that particular pixel. 
    Thresholding for a suitable value  of R gives the corners in the image.
    """

   

    R = np.zeros(greyscale_image.shape) 

    # Apply gaussian blur
    greyscale_image = cv2.GaussianBlur(greyscale_image, (window_size, window_size), 3)

    # Compute image gradients
    Ix = cv2.Sobel(greyscale_image, cv2.CV_64F, 1, 0, ksize=9)
    Iy = cv2.Sobel(greyscale_image, cv2.CV_64F, 0, 1, ksize=9)

    # Compute elements of the structure tensor
    # element wise product of Ix and Iy
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Compute sums of the structure tensor elements over the window
    #This step gives the product of the gradient components for the Matrix M.

    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 3)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 3)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 3)

    # Compute the corner response function
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy

    """
    M =[Ix^2  IxIy]
       [IxIy  Iy^2]
    R = det(M) - k * trace(M)^2

    The corner response matrix, denoted as R, is formed based on the gradient information obtained from the image. 
    It represents the likelihood of each pixel being a corner.
    When |R| is small, which happens when λ1 and λ2 are small, the region is flat.
    When R<0, which happens when λ1>>λ2 or vice-versa, the region is an edge
    When R is large, which happens when λ1 and λ2 are large and λ1∼λ2, the region is a corner
    """
   # Compute the corner response function
    R = det - k * (trace ** 2)

   # Determine the corners based on the threshold. 
    max_Value = (threshold/100) * np.max(R)
    # Create a copy of the original image to draw corners on
    corners_image = np.copy(original_image)

    # Iterate over each pixel in the image and draw a filled circle at the corner position
    # if the pixel value is greater than the threshold and is the maximum value in the 3x3 neighborhood
    for i in range(1, R.shape[0] - 1):
        for j in range(1, R.shape[1] - 1):
            if R[i, j] > max_Value and R[i, j] == np.max(R[i - 1:i + 2, j - 1:j + 2]):
                 cv2.circle(corners_image, (j, i), 5, (255, 0, 0), -1)  # Draw a filled circle at the corner position


    return corners_image





    
    
def HoughLine(img,numberOfLines,resolution):
    # Apply edge detection method on the image
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    height, width = edges.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # max_dist
    max_rho = img_diagonal
    
    #transformation from image space to parameter space
     # x cos(theta) + y sin(theta) = rho

    ### parameter space limits
    rhos = np.arange(-max_rho, max_rho + 1, resolution)
    #'rho' is the distance from the origin to the line along a vector perpendicular to the line. The range of possible 'rho' values is from -img_diagonal to img_diagonal.
    thetas = np.deg2rad(np.arange(0, 180, resolution))
    # thetas covers all possible orientations of the line
    print('thetas',thetas.shape)
    print('rhos',rhos.shape)
    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    x_canny, y_canny = np.nonzero(edges) # find all edge (nonzero) pixel indexes
    print('accumulator',accumulator.shape)
    print('y',y_canny.shape)
    print('x',x_canny.shape)
    for i in range(len(x)): # cycle through edge points
        x = x_canny[i]
        y = y_canny[i]
        

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int(x * np.cos(thetas[j]) + y * np.sin(thetas[j]))
            rho += int(img_diagonal) ##accounting for negative values
            accumulator[rho, j] += 1
            
    accumulator=  np.where(accumulator > numberOfLines, accumulator, 0)       
    # Find indices of non-zero values in thresholded accumulator array
    rho_idxs, theta_idxs = np.nonzero(accumulator)

    # Extract rho and theta values for detected lines
    rhos_detected = rhos[rho_idxs]
    thetas_detected = thetas[theta_idxs]

    # Combine rho and theta values into a single array of line parameters
    lines = np.column_stack((rhos_detected, thetas_detected))
    return lines
    
    
    

            
       
            
def draw_lines_on_image(img, lines):
    # Draw lines on a blank image
    #generate the x y vals from rho and theta values
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Convert original image to RGB if it's grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Overlay line image on original image
    overlay_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    return overlay_img   
    
    
    
    
    
    
    
    
    
    