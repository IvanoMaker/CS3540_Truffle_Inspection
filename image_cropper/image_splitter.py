import cv2
import numpy as np
import os

# ensure these two folders are in the same directory as this script
OUTPUT_DIR = "image_cropper/output"      # output directory for cropped images
INPUT_DIR = "image_cropper/input_images" # input directory for images

imgs = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]  # get list of image files in the input_images folders
print(f"Found {len(imgs)} images in {INPUT_DIR}...")                # "Found x images in input_images..."
print(f"Processing and saving cropped images to {OUTPUT_DIR}...")   # "Processing and saving cropped images to output..."

count = 0

# detect and crop truffles from each image
for k, img_file in enumerate(imgs):
    img = cv2.imread(os.path.join(INPUT_DIR, img_file))        # import image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # dark mask
    _, dark_mask = cv2.threshold(
        L, 135, 255, cv2.THRESH_BINARY_INV
    )

    # red mask
    _, red_mask = cv2.threshold(
        A, 150, 255, cv2.THRESH_BINARY
    )

    # Invert red mask so red becomes black (removed)
    red_mask_inv = cv2.bitwise_not(red_mask)

    # dark and not red
    mask = cv2.bitwise_and(dark_mask, red_mask_inv)

    # morphology tuned for round objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
    for cnt in contours:
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
        output_path = os.path.join(OUTPUT_DIR, f"truffle_{count}.png")
        cv2.imwrite(output_path, crop_img)
        count += 1
        
    print(f"{k+1}/{len(imgs)} images processed...")  # "x/y images processed..."
print("All images processsed.")                      # "All images processed."