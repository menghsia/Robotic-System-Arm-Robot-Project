#!/usr/bin/python3

"""Example:
ros2 launch realsense2_camera rs_l515_launch.py

python3 transform_rgb_depth.py

The goal of this example is to transform an RGB image and its corresponding depth image 
from an initial camera viewpoint (T_i) to a final camera viewpoint (T_f). 

This example will show the result in top view
"""

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import time


def transform_images(rgb_img, depth_img, K, T_i, T_f):
    """
    Transforms the RGB and depth images to a new viewpoint based on the provided camera extrinsic matrices.

    Parameters:
        rgb_img (np.array): Original RGB image.
        depth_img (np.array): Original depth image.
        K (np.array): Camera intrinsic matrix (3x3).
        T_i (np.array): Initial camera extrinsic matrix (4x4) representing the pose of the camera in the world frame.
        T_f (np.array): Final camera extrinsic matrix (4x4) representing the desired pose of the camera in the world frame.

    Returns:
        tuple: Transformed RGB image, Transformed depth image.
    """

    # Use to ensure that the entire transformed image is contained within the output
    scale_factor=1.2

    h, w = rgb_img.shape[:2]
    
    # Calculate the relative transformation matrix between the initial and final camera poses
    T_relative = np.dot(T_f, np.linalg.inv(T_i))
    
    # Compute the new projection matrix for the RGB image
    P_new = np.dot(K, T_relative[:3, :3])
    P_inv = np.linalg.inv(P_new)
    
    # Compute the homography for RGB image
    H_rgb = np.dot(P_new, np.linalg.inv(K))
    
    # Create a larger canvas for RGB
    enlarged_h_rgb, enlarged_w_rgb = int(h * scale_factor), int(w * scale_factor)
    
    # Warp the RGB image using the computed homography onto the larger canvas
    warped_rgb = cv2.warpPerspective(rgb_img, H_rgb, (enlarged_w_rgb, enlarged_h_rgb))
    
    # For the depth values, we first transform them to 3D points, apply the T_relative transformation, and then project them back to depth values
    # Back-project to 3D camera coordinates
    u = np.repeat(np.arange(w)[None, :], h, axis=0)
    v = np.repeat(np.arange(h)[:, None], w, axis=1)
    
    Z = depth_img
    X = (u - K[0,2]) * Z / K[0,0]
    Y = (v - K[1,2]) * Z / K[1,1]
    
    # Homogeneous coordinates in the camera frame
    points_camera_frame = np.stack((X, Y, Z, np.ones_like(Z)), axis=-1)
    
    # Apply the relative transformation to the depth points
    points_transformed = np.dot(points_camera_frame, T_relative.T)
    
    # Project back to depth values
    depth_transformed = points_transformed[..., 2]
    
    # Create a larger canvas for depth
    enlarged_h_depth, enlarged_w_depth = int(h * scale_factor), int(w * scale_factor)
    
    # Use the same homography as RGB for depth
    warped_depth = cv2.warpPerspective(depth_transformed, H_rgb, (enlarged_w_depth, enlarged_h_depth))
    
    return warped_rgb, warped_depth


class RGBSubscriber(Node):
    def __init__(self):
        super().__init__('rgb_subscriber_node')
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            10)
        self.rgb_subscription

    def rgb_callback(self, msg):
        rgb_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        self.rgb_img = rgb_img

class DepthSubscriber(Node):
    def __init__(self):
        super().__init__('depth_subscriber_node')
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
        self.depth_subscription

    def depth_callback(self, msg):
        depth_img = CvBridge().imgmsg_to_cv2(msg, "16UC1")
        self.depth_img = depth_img

def main():
    rclpy.init()
    rgb_subscriber = RGBSubscriber()
    depth_subscriber = DepthSubscriber()
    print("ROS Nodes started.")

    while rclpy.ok():
        rclpy.spin_once(rgb_subscriber)
        if rgb_subscriber.rgb_img is None:
            continue
        rclpy.spin_once(depth_subscriber)
        if depth_subscriber.depth_img is None:
            continue
       
        # Once rgb and depth frames are obtained
        rgb_img = rgb_subscriber.rgb_img
        depth_img = depth_subscriber.depth_img

        # Add the physical properties of your camera
        K = np.array([[904.5715942382812,   0.     , 635.9815063476562 ],
                     [0.     , 905.2954711914062, 353.06036376953125],
                     [0.     ,   0.     ,   1.     ]]) # Camera intrinsic matrix

        T_i = np.array([[ 9.99243787e-01,-3.00692904e-02,  2.46513928e-02,  3.67050953e+01],
                        [-3.38669044e-02, -9.84536238e-01,  1.71876203e-01,  1.23669566e+02],
                        [ 1.91019941e-02, -1.72581094e-01, -9.84810073e-01,  1.12152429e+03],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])

        T_f = np.array([
                        [1, 0,  0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 1000],   
                        [0, 0,  0, 1]
                        ]) # Final camera extrinsic matrix (desired pose of the camera in the world frame)

        # Transformation functions use here
        warped_rgb, warped_depth = transform_images(rgb_img, depth_img, K, T_i, T_f)

        # Assuming max_depth_value is the maximum depth value 
        max_depth_value = 1000   
        clipped_depth = np.clip(warped_depth, 0, max_depth_value)

        # Convert to 8-bit image to apply colormap
        scaled_depth = cv2.convertScaleAbs(clipped_depth, alpha=(255.0/max_depth_value))
        warped_depth_color = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)

        # Display images:
        cv2.imshow('Original RGB', rgb_img)
        # cv2.imshow('Original Depth', depth_img)
        cv2.imshow('Warped RGB', warped_rgb)
        cv2.imshow('Warped Depth', scaled_depth)
        cv2.imshow('Warped Depth Color Map', warped_depth_color)

        # Break loop on keyboard interruput (q)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Cleaning up 
    cv2.destroyAllWindows()
    rgb_subscriber.rgb_subscription.destroy()
    depth_subscriber.depth_subscription.destroy()
    rgb_subscriber.destroy_node()
    depth_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
