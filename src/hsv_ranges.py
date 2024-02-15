# This program allows the user to quickly find the hsv ranges for a colored object to create a mask.
# You can adjust the sliders in the HSV ranges and enter 'r' to show the lower and upper range arrays.
import cv2
import numpy as np
import time

# Method for trackbar function
def track(y):
    pass

# Finding all the cameras available. Check which video ID X in /dev/videoX is not being opened
# Checks the first 10 indices; may have to increase the index range for seeing more IDs
def returnCameraIndices():
    # checks the first 10 indices.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr
returnCameraIndices()

# Initializing camera feed
cam = cv2.VideoCapture(0)  # check which camera X from /dev/videoX is open from returnCameraIndices() array
cam.set(3, 1280)  # frame width
cam.set(4, 720)  # frame height

# Create a window named trackbars.
cv2.namedWindow("hsv sliders")

# Hue range: 0-179, Saturation range: 0-255, Value range: 0-255
cv2.createTrackbar("lower bound - H", "hsv sliders", 0, 179, track)
cv2.createTrackbar("lower bound - S","hsv sliders", 0, 255, track)
cv2.createTrackbar("lower bound - V", "hsv sliders", 0, 255, track)
cv2.createTrackbar("upper bound - H", "hsv sliders", 179, 179, track)
cv2.createTrackbar("upper bound - S", "hsv sliders", 255, 255, track)
cv2.createTrackbar("upper bound - V", "hsv sliders", 255, 255, track)
try:

    while True:
        ret, frame = cam.read()
        if not ret and not frame:
            break

        #  converting BGR image to HSV
        hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #  Trackbar values updated
        lh = cv2.getTrackbarPos("lower bound - H", "hsv sliders")
        ls = cv2.getTrackbarPos("lower bound - S","hsv sliders")
        lv = cv2.getTrackbarPos("lower bound - V", "hsv sliders")
        uh = cv2.getTrackbarPos("upper bound - H", "hsv sliders")
        us = cv2.getTrackbarPos("upper bound - S", "hsv sliders")
        uv = cv2.getTrackbarPos("upper bound - V", "hsv sliders")

        lower_hsvrange = np.array([lh, ls, lv])
        upper_hsvrange = np.array([uh, us, uv])

        # Binary mask with white as target color
        bw_mask = cv2.inRange(hsvframe, lower_hsvrange, upper_hsvrange)

        # Mask with target color
        tcol_mask = cv2.bitwise_and(frame, frame, mask=bw_mask)

        # 3 channel image mask from binary mask
        mask_ch3 = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)

        stacked_frames = np.hstack((tcol_mask, frame, mask_ch3))
        cv2.imshow('hsv sliders', cv2.resize(stacked_frames, None, fx=0.5, fy=0.5))

        key = cv2.waitKey(1)

        if key == 114:  # ASCII code for 'r' key
            ranges = [[lh, ls, lv], [uh, us, uv]]
            print(ranges)

        if key == 27:  # ASCII code for ESC key
            break

finally:
    cam.release()
    cv2.destroyAllWindows()