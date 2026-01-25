import cv2
import numpy as np
from typing import Tuple

def extract_contour_points(image_path: str, num_points: int = 500) -> np.ndarray:
    """
    Extracts ordered contour points from an image.
    
    Args:
        image_path: Path to the input image file.
        num_points: The number of points to sample from the contour.
        
    Returns:
        A numpy array of shape (num_points, 2) containing (x, y) coordinates.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Use thresholding to find the robot
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No contours found in the image.")

    # Get the largest contour
    main_contour = max(contours, key=cv2.contourArea)
    
    # Reshape and convert to float
    points = main_contour.reshape(-1, 2).astype(float)
    
    # Resample to the desired number of points
    total_len = len(points)
    indices = np.linspace(0, total_len - 1, num_points).astype(int)
    resampled_points = points[indices]
    
    # Center the points
    centroid = np.mean(resampled_points, axis=0)
    resampled_points -= centroid
    
    # Flip Y axis (fix upside-down issue)
    resampled_points[:, 1] = -resampled_points[:, 1]
    
    return resampled_points
