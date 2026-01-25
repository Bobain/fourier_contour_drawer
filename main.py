import os
import sys
import numpy as np

from src.contour import extract_contour_points
from src.fourier import compute_dft
from src.animator import FourierDrawerAnimator

def select_image(input_dir: str) -> str:
    """
    Scans the input directory for images and asks the user to select one.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"Error: No valid images found in {input_dir}")
        sys.exit(1)
        
    print("\nAvailable images in 'input/' folder:")
    for i, img in enumerate(images, 1):
        print(f"{i}. {img}")
        
    while True:
        try:
            choice = int(input(f"\nSelect an image (1-{len(images)}): "))
            if 1 <= choice <= len(images):
                return os.path.join(input_dir, images[choice - 1])
            else:
                print(f"Please enter a number between 1 and {len(images)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """
    Main orchestration function for the Fourier Drawing Machine.
    """
    input_dir = "input"
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Fourier Drawing Machine Initializing ---")
    
    image_path = select_image(input_dir)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_fourier_contour.gif")
    
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
    # Ask the user for the number of circles
    max_circles = min(len(x_coeffs), len(y_coeffs))
    while True:
        try:
            num_circles_input = input(f"\nHow many circles would you like to use? (max {max_circles}, default 100): ").strip()
            num_circles = int(num_circles_input) if num_circles_input else 100
            if 1 <= num_circles <= max_circles:
                break
            else:
                print(f"Please enter a number between 1 and {max_circles}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Ask for number of frames
    while True:
        try:
            frames_input = input("\nHow many frames for the animation? (default 300, fewer = smaller file): ").strip()
            num_frames = int(frames_input) if frames_input else 300
            if num_frames > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Ask for compact mode
    is_compact = input("\nEnable Compact Mode? (Smaller resolution, reduces file size) [y/N]: ").lower() == 'y'
    figsize = (6, 6) if is_compact else (10, 10)

    print(f"Initializing animator with top {num_circles} circles per component...")
    animator = FourierDrawerAnimator(
        image_path=image_path,
        x_coeffs=x_coeffs[:num_circles],
        y_coeffs=y_coeffs[:num_circles],
        original_points=points,
        figsize=figsize
    )
    
    print(f"Generating GIF ({num_frames} frames)...")
    animator.save_animation(output_path, frames=num_frames)
    
    print(f"Success! Animation saved to: {output_path}")

if __name__ == "__main__":
    main()
