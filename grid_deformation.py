import os
import cv2
import numpy as np

# Set the folder containing extracted images
image_folder = "test_images_1/filtered_images"
output_folder = "test_images_1/output_images_filtered"
video_filename = "grid_deformation.mp4"

os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])  # Sort images in order

if len(image_files) < 2:
    raise ValueError("Need at least two images to track displacement!")

# Load the first image and convert to grayscale
first_img = cv2.imread(os.path.join(image_folder, image_files[0]))
gray_first = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
height, width, _ = first_img.shape  # Get image dimensions

# ** Step 1: Create a Grid of Points **
grid_size = 10  # Grid spacing
rows, cols = gray_first.shape
grid_points = np.array([[x, y] for y in range(0, rows, grid_size) for x in range(0, cols, grid_size)], dtype=np.float32)
grid_points = grid_points.reshape(-1, 1, 2)  # Ensure correct shape for OpenCV

# ** Step 2: Draw Initial Grid Without Displacement **
initial_grid_img = first_img.copy()
for pt in grid_points:
    x, y = int(pt[0][0]), int(pt[0][1])
    cv2.circle(initial_grid_img, (x, y), 1, (0, 255, 0), -1)  # Green dots for grid points

# Save initial frame with grid points
first_frame_path = os.path.join(output_folder, "frame_0000.png")
cv2.imwrite(first_frame_path, initial_grid_img)

frame_count = 1  # Start counting from 1 since 0 is the initial grid frame

# ** Step 3: Track Grid Points Using Optical Flow **
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

for i in range(1, len(image_files)):
    next_img = cv2.imread(os.path.join(image_folder, image_files[i]))
    gray_next = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    # Track the grid points
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_first, gray_next, grid_points, None, **lk_params)

    # Ensure only successfully tracked points are used
    if next_pts is None or status is None or len(next_pts) == 0:
        print(f"⚠ Warning: No points tracked in frame {i}. Skipping...")
        continue

    good_old = grid_points[status.flatten() == 1]
    good_new = next_pts[status.flatten() == 1]

    # ** Step 4: Deform and Overlay the Grid **
    deformed_grid_img = next_img.copy()

    for old, new in zip(good_old, good_new):
        x_old, y_old = old.ravel()  # Unpack coordinates
        x_new, y_new = new.ravel()

        # Draw deformed grid points
        cv2.circle(deformed_grid_img, (int(x_new), int(y_new)), 1, (0, 255, 0), -1)  # Green dots
        cv2.line(deformed_grid_img, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 0, 255), 1)  # Red lines

    # Save the deformed frame
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, deformed_grid_img)
    frame_count += 1

    # Show the frame (optional)
    cv2.imshow(f"Deformed Grid - Frame {i}", deformed_grid_img)
    cv2.waitKey(100)  # Display for 100ms per frame

    # Update for next iteration
    gray_first = gray_next
    grid_points = good_new.reshape(-1, 1, 2)  # Update grid points for tracking

cv2.destroyAllWindows()

# ** Step 5: Create Video from Frames **
frame_rate = 5  # Frames per second
video_path = os.path.join(output_folder, video_filename)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

# Add each frame to the video
for frame in sorted(os.listdir(output_folder)):
    if frame.endswith(".png"):
        frame_img = cv2.imread(os.path.join(output_folder, frame))
        video_writer.write(frame_img)

video_writer.release()
print(f"🎥 Video saved at: {video_path}")
