import cv2
import numpy as np
import SimpleITK as sitk

def lucas_kanade_registration(img1_path, img2_path):
    """
    Registers img1 with respect to img2 using Lucas-Kanade Optical Flow.
    Returns the registered image and the flow visualization.
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Detect good features to track
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

    # Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

    # Compute transformation matrix
    p0 = p0[st == 1]
    p1 = p1[st == 1]
    if len(p0) < 4:  # Not enough points for transformation
        return None, None
    transform_matrix, _ = cv2.estimateAffinePartial2D(p0, p1)

    # Apply transformation
    registered_img = cv2.warpAffine(img1, transform_matrix, (img2.shape[1], img2.shape[0]))

    return registered_img

def farneback_registration(img1_path, img2_path):
    """
    Registers img1 with respect to img2 using Farneback Optical Flow.
    Returns the registered image and the flow visualization.
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Warp image using flow field
    h, w = img1.shape
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h))) + flow.reshape(-1, 2)
    remapped_img = cv2.remap(img1, flow_map.astype(np.float32), None, cv2.INTER_LINEAR)

    return remapped_img

def sift_registration(img1_path, img2_path):
    """
    Registers img1 with respect to img2 using SIFT keypoints and Homography.
    Returns the registered image.
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # SIFT detector
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match features
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Compute transformation
    if len(good_matches) < 4:  # Not enough points for transformation
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image
    registered_img = cv2.warpPerspective(img1, homography_matrix, (img2.shape[1], img2.shape[0]))

    return registered_img

def demons_registration(img1_path, img2_path):
    """
    Registers img1 with respect to img2 using Demons Registration.
    Returns the registered image.
    """
    img1 = sitk.ReadImage(img1_path, sitk.sitkFloat32)
    img2 = sitk.ReadImage(img2_path, sitk.sitkFloat32)

    # Apply Demons Registration
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(100)
    displacement_field = demons.Execute(img1, img2)

    # Warp image using displacement field
    registered_img = sitk.Warp(img1, displacement_field)

    # Normalize to 8-bit and convert back to SimpleITK Image
    registered_img_array = sitk.GetArrayFromImage(registered_img)
    registered_img_array = (255 * (registered_img_array - registered_img_array.min()) /
                            (registered_img_array.max() - registered_img_array.min())).astype(np.uint8)
    registered_img = sitk.GetImageFromArray(registered_img_array)

    return registered_img

if __name__ == "__main__":
    img1_path = "test_images_1/filtered_images/0.png"
    img2_path = "test_images_1/filtered_images/1.png"

    # Run different registration methods
    lk_reg = lucas_kanade_registration(img1_path, img2_path)
    fb_reg = farneback_registration(img1_path, img2_path)
    sift_reg = sift_registration(img1_path, img2_path)
    demons_reg = demons_registration(img1_path, img2_path)

    # Save results
    if lk_reg is not None:
        cv2.imwrite("lucas_kanade_registered.png", lk_reg)
    if fb_reg is not None:
        cv2.imwrite("farneback_registered.png", fb_reg)
    if sift_reg is not None:
        cv2.imwrite("sift_registered.png", sift_reg)
    if demons_reg is not None:
        sitk.WriteImage(demons_reg, "demons_registered.png")

    print("All registration algorithms executed successfully!")
