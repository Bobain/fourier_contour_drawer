import os
import sys
import numpy as np

# Ensure src is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from contour import extract_contour_points
from fourier import compute_dft
from animator import FourierDrawerAnimator

def main():
    """
    Main orchestration function for the Fourier Drawing Machine.
    """
    image_path = "nono_le_petit_robot.png"
    output_dir = "output"
    output_path = os.path.join(output_dir, "nono_fourier_machine.gif")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Fourier Drawing Machine Initializing ---")
    
    # 1. Extract Points
    print("Extracting contour from image...")
    try:
        points = extract_contour_points(image_path, num_points=600)
    except Exception as e:
        print(f"Error extracting contour: {e}")
        return

    # 2. Compute Fourier Coefficients for X and Y separately
    print("Computing Fourier coefficients...")
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    x_coeffs = compute_dft(x_coords)
    y_coeffs = compute_dft(y_coords)
    
    # 3. Create Animation
    # Limit number of circles for better performance and visual clarity
    num_circles = 100
    print(f"Initializing animator with top {num_circles} circles per component...")
    animator = FourierDrawerAnimator(
        image_path=image_path,
        x_coeffs=x_coeffs[:num_circles],
        y_coeffs=y_coeffs[:num_circles],
        original_points=points
    )
    
    print("Generating GIF (this may take a minute)...")
    animator.save_animation(output_path, frames=300)
    
    print(f"Success! Animation saved to: {output_path}")

if __name__ == "__main__":
    main()
