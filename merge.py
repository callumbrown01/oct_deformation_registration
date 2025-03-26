import cv2
import os
import numpy as np

# Define input directories
filtered_dir = "filtered_images"
flow_vis_dir = "flow_vis"
warped_dir = "warped"

# Define output directory
output_dir = "merged_outputs"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

def merge_images(filtered_path, flow_path, warped_path, output_path):
    """Merge three images horizontally and save the result."""
    img1 = cv2.imread(filtered_path)
    img2 = cv2.imread(flow_path)
    img3 = cv2.imread(warped_path)

    # Ensure images are loaded correctly
    if img1 is None or img2 is None or img3 is None:
        print(f"Skipping {filtered_path} - One or more images not found.")
        return

    # Resize all images to the same height (smallest height among them)
    min_height = min(img1.shape[0], img2.shape[0], img3.shape[0])
    img1 = cv2.resize(img1, (img1.shape[1], min_height))
    img2 = cv2.resize(img2, (img2.shape[1], min_height))
    img3 = cv2.resize(img3, (img3.shape[1], min_height))

    # Concatenate images horizontally
    merged_img = np.hstack((img1, img2, img3))

    # Save the merged image
    cv2.imwrite(output_path, merged_img)
    print(f"Saved merged image: {output_path}")

# Process images
for filename in os.listdir(filtered_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        filtered_path = os.path.join(filtered_dir, filename)
        flow_path = os.path.join(flow_vis_dir, filename)
        warped_path = os.path.join(warped_dir, filename)
        output_path = os.path.join(output_dir, filename)

        merge_images(filtered_path, flow_path, warped_path, output_path)

print("Merging complete! Merged images are saved in 'merged_outputs/'.")
