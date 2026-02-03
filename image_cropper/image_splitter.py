import cv2
import numpy as np
import os

# ensure these two folders are in the same directory as this script
OUTPUT_DIR = "output"      # output directory for cropped images
INPUT_DIR = "input_images" # input directory for images
#test: works on other computer

imgs = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]  # get list of image files in the input_images folders
print(f"Found {len(imgs)} images in {INPUT_DIR}...")                # "Found x images in input_images..."
print(f"Processing and saving cropped images to {OUTPUT_DIR}...")   # "Processing and saving cropped images to output..."

# detect and crop truffles from each image
for k, img_file in enumerate(imgs):
    img = cv2.imread(os.path.join(INPUT_DIR, img_file))        # import image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert to LAB color space (https://en.wikipedia.org/wiki/CIELAB_color_space)
    L, A, B = cv2.split(lab)                    # split the LAB image to different channels

    # Thresholding to create a binary mask for dark regions (truffles)
    # for each pixel in L channel, if value < 120, set to 255 (white), else set to 0 (black)
    _, dark_mask = cv2.threshold(L, 120, 255, cv2.THRESH_BINARY_INV)

    # create kernal for morphological operations
    # use closing to fill small holes, then opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # find contours in the mask, this is the detecting part
    contours, _ = cv2.findContours(
        mask,                   # use the mask defined above
        cv2.RETR_EXTERNAL,      # retrieve only the external contours
        cv2.CHAIN_APPROX_SIMPLE # compress horizontal, vertical, and diagonal segments and leave only their end points
    )

    # estimate single-truffle radius from data
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 2000]
    median_area = np.median(areas)
    SINGLE_RADIUS = int(np.sqrt(median_area / np.pi))
    CROP_RADIUS = int(SINGLE_RADIUS * 1.2)  # margin

    # for each controur found
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 2000: # filter out small areas, like drops from the icing
            continue

        M = cv2.moments(cnt) # calculate moments for each contour
        if M["m00"] == 0:    # avoid division by zero errors
            continue

        cx = int(M["m10"] / M["m00"]) # centroid x
        cy = int(M["m01"] / M["m00"]) # centroid y

        r = CROP_RADIUS

        # crop square dimensions
        y1 = max(0, cy - r)
        y2 = min(img.shape[0], cy + r)
        x1 = max(0, cx - r)
        x2 = min(img.shape[1], cx + r)

        # crop the image
        crop_img = img[y1:y2, x1:x2]
        # skip empty crops
        if crop_img.size == 0:
            continue
        # save cropped image to the directory
        output_path = os.path.join(OUTPUT_DIR, f"truffle_{i}.png")
        cv2.imwrite(output_path, crop_img)
    print(f"{k+1}/{len(imgs)} images processed...")  # "x/y images processed..."
print("All images processsed.")                      # "All images processed."