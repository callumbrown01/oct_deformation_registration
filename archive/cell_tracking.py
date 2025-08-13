import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

def load_images(image_paths, verbose=False):
    """Load grayscale images from given file paths."""
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_paths]
    if verbose:
        print(f"Loaded {len(images)} images.")
    return images

def compute_optical_flow(base_img, deformed_img, verbose=False):
    """Compute dense optical flow using the Farneback method."""
    flow = cv2.calcOpticalFlowFarneback(base_img, deformed_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    displacement_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    if verbose:
        print("Computed optical flow.")
    return flow, displacement_magnitude

def warp_image_using_flow(image, flow, verbose=False):
    """Use the optical flow field to warp an image."""
    h, w = flow.shape[:2]
    
    # Generate a grid of coordinates (h, w, 2)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')  # Ensure correct shape
    flow_map = np.stack((x_coords, y_coords), axis=-1).astype(np.float32)  # Shape (h, w, 2)

    # Add optical flow displacement
    flow_map += flow.astype(np.float32)

    # Ensure image is float32 for remapping
    warped_image = cv2.remap(image, flow_map[..., 0], flow_map[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if verbose:
        print("Warped baseline image to align with deformed image.")

    return warped_image

def measure_tracking_accuracy(original, transformed, verbose=False):
    """Measure tracking accuracy using cross-correlation."""
    correlation = correlate2d(original, transformed, mode='same', boundary='symm')
    max_corr = np.max(correlation)
    normalized_corr = max_corr / (np.std(original) * np.std(transformed) * original.size)
    
    if verbose:
        print(f"Tracking Accuracy: {normalized_corr:.4f}")
    
    return normalized_corr

def visualize_displacement(flow, save_path=None, verbose=False):
    """Visualize the optical flow as arrows."""
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:10, 0:w:10]  # Sample every 10 pixels
    fx, fy = flow[y, x].T

    plt.figure(figsize=(8, 6))
    plt.imshow(np.zeros_like(flow[..., 0]), cmap='gray')
    plt.quiver(x, y, fx, fy, color='r', angles='xy', scale_units='xy', scale=1)
    plt.title("Optical Flow Displacement")

    if save_path:
        plt.savefig(save_path)
        if verbose:
            print(f"Displacement visualization saved as {save_path}")

# Example Usage
def main(image_paths, verbose=False):
    images = load_images(image_paths, verbose)
    base_img = images[0]

    for i, deformed_img in enumerate(images[1:], start=1):
        # Compute optical flow
        flow, displacement = compute_optical_flow(base_img, deformed_img, verbose)

        # Warp the baseline image
        warped_img = warp_image_using_flow(base_img, flow, verbose)

        # Measure accuracy
        accuracy = measure_tracking_accuracy(deformed_img, warped_img, verbose)

        # Save results
        cv2.imwrite(f"warped_image_{i}.png", warped_img)
        if verbose:
            print(f"Warped image saved as warped_image_{i}.png")

        visualize_displacement(flow, save_path=f"flow_visualization_{i}.png", verbose=verbose)

# Paths to images (replace with actual file paths)
image_paths = ["test_images_1/filtered_images/0.png", "test_images_1/filtered_images/1.png", "test_images_1/filtered_images/2.png", "test_images_1/filtered_images/3.png", "test_images_1/filtered_images/4.png", "test_images_1/filtered_images/5.png", "test_images_1/filtered_images/6.png"]  # Replace with actual paths

# Run the program with verbosity enabled
main(image_paths, verbose=True)

