#!/usr/bin/env python
""" Example: 

python3 save_from_realsense.py

Press 's' to save both rgd and colored depth image

Note: run ./launch_realsense_apriltag.sh first
"""
from __future__ import print_function

import sys
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

global cv_image
cv_image = None
global color_depth
color_depth = None
global img_counter
img_counter = 1

class ImageListener(Node):
    def __init__(self, topic):
        global cv_image
        global img_counter
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)

    def callback(self,data):
        global cv_image
        global img_counter
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        except CvBridgeError as e:
            self.get_logger().info(str(e))
 
        cv2.imshow("Image window", cv_image)
        k = cv2.waitKey(1) 
        if k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('saved_rgb_image'+str(img_counter)+'.png',cv_image)
            cv2.imwrite('saved_depth_image'+str(img_counter)+'.png',color_depth)
            img_counter += 1
            print("rgb/depth images saved!")


class DepthListener(Node):
    def __init__(self, topic):
        global color_depth
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)

    def callback(self,data):
        global color_depth
        global img_counter
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            self.get_logger().info(str(e))

        clipped = np.clip(cv_depth, 0, 2000).astype(np.uint8)
        normed = cv2.normalize(clipped, clipped, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_depth = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        cv2.imshow("Depth window", color_depth)
        k = cv2.waitKey(1) 
        if k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('saved_rgb_image'+str(img_counter)+'.png',cv_image)
            cv2.imwrite('saved_depth_image'+str(img_counter)+'.png',color_depth)
            img_counter += 1
            print("rgb/depth images saved!")


def main(args=None):
    rclpy.init(args=args)

    image_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"

    cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

    depth_listener = DepthListener(depth_topic)
    image_listener = ImageListener(image_topic)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(image_listener)
    executor.add_node(depth_listener)
    try:
      executor.spin()
    except KeyboardInterrupt:
      pass

    image_listener.destroy_node()
    depth_listener.destroy_node()

    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()