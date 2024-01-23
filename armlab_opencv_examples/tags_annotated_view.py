#!/usr/bin/env python
"""Example: 

python3 tags_annotated_view.py

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
from apriltag_msgs.msg import *

class ImageListener(Node):
    def __init__(self, topic,draw):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.draw = draw

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.draw.image = cv_image.copy()
        except CvBridgeError as e:
            self.get_logger().info(str(e))


class TagDetectionListener(Node):
    def __init__(self, topic, draw):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.draw = draw

    def callback(self, msg):
        self.draw.detections = msg
        self.draw.drawDetections()


class Draw():
    def __init__(self):
        self.image = np.zeros((720,1280, 3)).astype(np.uint8)
        self.detections = AprilTagDetectionArray()
    
    def drawDetections(self):
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        if np.any(self.image != 0):
            for detection in self.detections.detections:
                # draw center of the tags
                point = (int(detection.centre.x), int(detection.centre.y))
                cv2.circle(self.image, point, 3, GREEN, -1)

                # draw edges of the tags
                corners = detection.corners
                for i in range(4):
                    cv2.line(self.image, 
                             (int(corners[i].x), int(corners[i].y)),
                             (int(corners[(i+1)%4].x), int(corners[(i+1)%4].y)),
                            BLUE, 
                            3)
                    
                # draw the ID of the tags
                id = str(detection.id)
                cv2.putText(self.image, "ID:"+id, (int(corners[2].x), int(corners[2].y)-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA) 


            cv2.imshow("Tag View", self.image)
            cv2.waitKey(1) 
 

def main(args=None):
    rclpy.init(args=args)

    image_topic = "/camera/color/image_raw"
    tag_detection_topic = "/detections"

    cv2.namedWindow("Tag View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tag View", 1280, 720)

    draw = Draw()
    tag_listener = TagDetectionListener(tag_detection_topic, draw)
    image_listener = ImageListener(image_topic, draw)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(image_listener)
    executor.add_node(tag_listener)
    try:
      executor.spin()
    except KeyboardInterrupt:
      pass

    image_listener.destroy_node()
    tag_listener.destroy_node()

    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()