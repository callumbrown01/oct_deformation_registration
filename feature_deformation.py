import os
import cv2
import numpy as np

# Set paths
image_folder = "test_images_1/filtered_images"
output_folder = "test_images_1/output_images_edges"

os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])  # Sort images

if len(image_files) == 0:
    raise ValueError("No images found in the directory!")

for i, image_file in enumerate(image_files):
    img = cv2.imread(os.path.join(image_folder, image_file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # ** Apply Gaussian Blur to reduce noise **
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)  # Smooth image

    # ** Apply Canny Edge Detection (higher thresholds for strong edges) **
    edges = cv2.Canny(blurred, 150, 300)  # Detect only clear edges

    # ** Overlay edges in green over the original image **
    edge_overlay = img.copy()
    edge_overlay[edges > 0] = [0, 255, 0]  # Color edges green

    # Save the processed image
    output_path = os.path.join(output_folder, f"edges_{i:04d}.png")
    cv2.imwrite(output_path, edge_overlay)

    print(f"✅ Processed: {image_file} → {output_path}")

print(f"🎉 Edge overlay completed! Check the folder: {output_folder}")
