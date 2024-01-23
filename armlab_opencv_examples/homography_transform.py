#!/usr/bin/python
""" Example: 

python3 homography_transform.py -i chess.jpg

"""
import argparse
import sys
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the rgb image")
args = vars(ap.parse_args())

# Read image
image = cv2.imread(args["image"])

# Select source points to apply the homography transform from
src_pts = np.array([391, 100, 14, 271, 347, 624, 747, 298]).reshape((4,2))

# Select destination points to apply the homography transform to
dest_pts = np.array([100, 100, 
                100, 650,
                650, 650,
                650, 100,]).reshape((4, 2))

H = cv2.findHomography(src_pts, dest_pts)[0]

new_img = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

# Draw green dots to represent the source points
for pt in src_pts:
    cv2.circle(image, tuple(pt), 5, (0, 255, 0), -1)

# Draw red dots to represent the destination points
for pt in dest_pts:
    cv2.circle(new_img, tuple(pt), 5, (0, 0, 255), -1)

# Save image
cv2.imwrite("homography_transform.png", new_img)
cv2.imwrite("homography_transform_original.png", image)

