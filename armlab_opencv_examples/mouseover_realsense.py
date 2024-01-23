#!/usr/bin/env python
""" Example: 

python3 mouseover_realsense.py  

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

font = cv2.FONT_HERSHEY_SIMPLEX
avg_radius = 10

class ImageListener(Node):
    def __init__(self, topic, depth_data, position):
        super().__init__('image_listener')
        self.topic = topic
        self.depth_data = depth_data
        self.position = position
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        except CvBridgeError as e:
            self.get_logger().info(str(e))

        n, d = 0.0, 0.0
        for i in range(-avg_radius, avg_radius, 1):
            for j in range(-avg_radius, avg_radius, 1):
                y_index = self.position["y"]+j
                x_index = self.position["x"]+i
                if (0 <= x_index < self.depth_data[0].shape[1]) and (0 <= y_index < self.depth_data[0].shape[0]):
                    d += self.depth_data[0][y_index][x_index]
                    n += 1.0
        d = d/n if n > 0 else 0
        output_uvd = "u:%d, v:%d, d:%.2f" % (self.position["x"], self.position["y"], d)
        cv2.putText(cv_image, output_uvd, (10, 20), font, 0.5, (0,0,0))
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)


class DepthListener(Node):
    def __init__(self, topic, depth_data):
        super().__init__('depth_listener')
        self.topic = topic
        self.depth_data = depth_data
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)

    def callback(self,data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            self.get_logger().info(str(e))

        self.depth_data[0] = cv_depth.copy()
        clipped = np.clip(cv_depth, 0, 2000).astype(np.uint8)
        normed = cv2.normalize(clipped, clipped, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_depth = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        cv2.imshow("Depth window", color_depth)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    image_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"

    position = {"x": 0, "y": 0}
    def mouse_callback(event, x, y, flags, param):
        position["x"] = x
        position["y"] = y
        
    shared_depth_data = [np.zeros((720, 1280), dtype=np.uint16)]  # Shared mutable data

    cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image window", mouse_callback)

    depth_listener = DepthListener(depth_topic, shared_depth_data)
    image_listener = ImageListener(image_topic, shared_depth_data, position)

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