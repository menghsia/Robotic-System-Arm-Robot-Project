#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import pdb
import math


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)

        self.BlockContourImg = np.zeros((720,1280, 3)).astype(np.uint8)

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.dropFlag = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.grid_points_flattened = np.vstack([self.grid_points[0].ravel(), self.grid_points[1].ravel()])

        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        self.tag_ids_centers = {}  # dictionary initialized
        self.tag_ids_centers_corners = {}  # center, corners 1, 2, 3, 4

        self.cam_homography_matrix = np.array([])
        self.cam_extrinsic_maxtrix = 0
        self.world_coord_calib_flag = False

        
        self.colors = list((
        {'id': 'red', 'color': 112, 'range': (115,135)},
        {'id': 'orange', 'color': 110, 'range': (100,114)},
        {'id': 'yellow', 'color': 90, 'range': (70,99)},
        {'id': 'green', 'color': 50, 'range': (43,69)},
        {'id': 'blue', 'color': 30, 'range': (16,42)},
        {'id': 'violet', 'color': 140, 'range': (0,15)})
        )

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # world coordinates from control_station.py
        self.cs_x = None
        self.cs_y = None
        self.cs_z = None

        # for getting image after applying homography
        self.warped_img = None
        self.modified_warped_img_flag = False

        self.T_f = np.array([
                        [1, 0,  0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 1000],   
                        [0, 0,  0, 1]
                        ])  #final camera extrinsic
        
        self.K = np.array([[904.6,0,635.982],[0,905.29,353.06],[0,0,1]]) 

        self.firstbdflag = True
        self.BLOCKS=None

    def is_square(self, contour, eps=0.05):
        # Calculate the perimeter of the contour
        peri = cv2.arcLength(contour, True)
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        print("approx",approx)
        print("len:",len(approx))
        
        # Check if the polygon has 4 sides
        if len(approx) == 4:
            # Calculate the lengths of the sides
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            
            # Check if sides are approximately equal and the shape is close to a square
            if 0.85 <= aspectRatio <= 1.15: # Adjust this range as needed
                return True
        return False


    def retrieve_area_color(self, data, contour, labels):
        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        img_hsv=cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        mean = cv2.mean(img_hsv, mask=mask)[0]  #mean of hue
        find_color=False
        
        for label in labels:
            color_range=label["range"]
            low,high=color_range[0],color_range[1]
            if (low<=mean) and (mean<=high):
                find_color=True
                return label["id"]
        if not find_color:
            min_dist = (np.inf, None)
            for label in labels:
                d=(label["color"]-mean)*(label["color"]-mean)
                if d < min_dist[0]:
                    min_dist = (d, label["id"])
            return min_dist[1]


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)
        
        # if not (self.cam_homography_matrix.size == 0): 
        #     self.DepthFrameRGB = cv2.warpPerspective(self.DepthFrameRGB, self.cam_homography_matrix, (self.DepthFrameRGB.shape[1], self.DepthFrameRGB.shape[0]), flags=cv2.INTER_LINEAR)


    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

        # if not (self.cam_homography_matrix.size == 0):        
        #     self.DepthFrameRaw = cv2.warpPerspective(self.DepthFrameRaw, self.cam_homography_matrix, (self.DepthFrameRaw.shape[1], self.DepthFrameRaw.shape[0]), flags=cv2.INTER_LINEAR)


    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        #print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """

        # if self.modified_warped_img_flag:
        #     print(type(self.warped_img))
        #     rgb_image = self.warped_img #NDArray[uint8]
        #     cnt_image = self.warped_img
        
        # else:
        intrinsicMat = np.array([[904.6,0,635.982],[0,905.29,353.06],[0,0,1]])     # factory intrinsic matrix
        K_inv = np.linalg.inv(intrinsicMat)
        self.BLOCKS=None
        # self.BLOCKS=np.zeros((15,6))
        # count=0


        if self.firstbdflag:
            time.sleep(1)
            self.firstbdflag = False

        rgb_image = self.VideoFrame.copy() #NDArray[uint8]
        cnt_image = self.VideoFrame.copy()
        
        depth_data = self.DepthFrameRaw.copy() 
        h, w = rgb_image.shape[:2]

        #TODO!!! Multiply PnP-solved extrinsic matrix by a positive 2 degree rotation homogeneous matrix!!
        # T_i= np.array([[ 9.99260666e-01, -3.84231535e-02,  1.33479174e-03,  3.05227949e+01],
        # [-3.78616658e-02, -9.77439597e-01,  2.07793954e-01,  1.10163653e+02],
        # [-6.67942069e-03, -2.07690863e-01, -9.78171708e-01,  1.03705217e+03],
        # [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        desired_matrix=np.array([[ 9.99260666e-01, -3.84231535e-02,  1.33479174e-03,  3.05227949e+01],
        [-3.78616658e-02, -9.77439597e-01,  2.07793954e-01,  1.10163653e+02],
        [-6.67942069e-03, -2.07690863e-01, -9.78171708e-01,  1.03705217e+03],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        
        if self.world_coord_calib_flag:
            T_i = self.cam_extrinsic_maxtrix 
            # T_i=np.dot(T_i,np.array([[1,0,0,0],[0,0.99939083,-0.0348995,0],[0,0.0348995,0.99939083,0],[0,0,0,1]]))  #old one
            # T_i=np.dot(T_i,np.array([[1.0, 2.1954964e-18, 1.35375648e-19, 3.55271368e-15],
            #[-2.46374145e-11, 0.999390827, -0.      034899497, 3.69089278e-07],
            #[1.62571546e-11, 0.0348994971, 0.999390827, 5.41551799e-08],[0,0,0,1.0]]))
            transform_matrix=np.array(np.dot(desired_matrix,np.linalg.inv(T_i)))

            T_i=np.dot(transform_matrix,T_i)
            # print(T_i)
            

        else:
            T_i = np.array([[1,0,0,0],[0,-0.9797,0,190],[0,0.2004,-0.9797,970],[0,0,0,1]])

        T_relative = np.dot(self.T_f, np.linalg.inv(T_i)) # Calculate the relative transformation matrix between the initial and final camera poses
        u = np.repeat(np.arange(w)[None, :], h, axis=0)
        v = np.repeat(np.arange(h)[:, None], w, axis=1)
        Z = depth_data
        X = (u - self.K[0,2]) * Z / self.K[0,0]
        Y = (v - self.K[1,2]) * Z / self.K[1,1]

        # Homogeneous coordinates in the camera frame
        points_camera_frame = np.stack((X, Y, Z, np.ones_like(Z)), axis=-1)

        # Apply the relative transformation to the depth points
        points_transformed = np.dot(points_camera_frame, T_relative.T)
        
        # Project back to depth values
        depth_data = points_transformed[..., 2]

        if self.world_coord_calib_flag:      
            depth_data = cv2.warpPerspective(depth_data, self.cam_homography_matrix, (w, h), flags=cv2.INTER_LINEAR)
            # print("depth:",depth_data)

    
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.rectangle(mask, (130,14),(1152,703), 255, cv2.FILLED)  # board box
        cv2.rectangle(mask, (562,361),(720,720), 0, cv2.FILLED)  # arm box

        cv2.rectangle(cnt_image, (120,14),(1164,703), (255, 0, 0), 2)  # board box
        cv2.rectangle(cnt_image, (562,361),(725,720), (255, 0, 0), 2)  # arm box

        thresh = cv2.bitwise_and(cv2.inRange(depth_data, 900, 990), mask)
        # cv2.imshow("Block detections window", thresh)  # update rate??
        # thresh = cv2.bitwise_and(cv2.inRange(depth_data, 500, 960), mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)

        for contour in contours:  
            M = cv2.moments(contour)
            color = self.retrieve_area_color(rgb_image, contour, self.colors)
            theta = cv2.minAreaRect(contour)[2]
            rad= int(theta) * (math.pi / 180)
            if M["m00"] >200: #filter noise
                # if self.is_square(contour): #only detect squares
                    
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                cv2.putText(cnt_image, str(int(cx)), (cx+30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
                # cv2.putText(cnt_image, str(int(cy)), (cx+20, cy+40), self.font, 1.0, (0,0,0), thickness=2)
                cv2.putText(cnt_image, color, (cx-30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
                cv2.putText(cnt_image, str(int(theta)), (cx, cy), self.font, 0.5, (255,255,255), thickness=2)

                self.BlockContourImg = cnt_image
                cv2.imshow("Block detections window", cv2.cvtColor(self.BlockContourImg, cv2.COLOR_RGB2BGR))  # update rate??
                if color=="red":
                    colornum=0
                if color=="orange":
                    colornum=1
                if color=="yellow":
                    colornum=2
                if color=="green":
                    colornum=3
                if color=="blue":
                    colornum=4
                if color=="violet":
                    colornum=5
                
                if (self.cam_homography_matrix.size != 0): #if calibrated
                    extrinsicMat = self.cam_extrinsic_maxtrix
                    homography_mat = self.cam_homography_matrix
                    point_uv1 = np.array([int(cx),int(cy),1])
                    point_uv_dash = np.dot(np.linalg.inv(homography_mat), point_uv1)
                    point_uv_dash[0] = point_uv_dash[0]/point_uv_dash[2]
                    point_uv_dash[1] = point_uv_dash[1]/point_uv_dash[2]
                    point_uv_dash[2] = point_uv_dash[2]/point_uv_dash[2]
                    depth_camera = self.DepthFrameRaw[int(point_uv_dash[1])][int(point_uv_dash[0])]
                    point_camera = depth_camera * (np.dot(K_inv,point_uv_dash))
                    point_camera=np.append(point_camera, [1])
                    points_transformed_ideal = np.dot(np.linalg.inv(extrinsicMat), point_camera)
                    xyz_w = points_transformed_ideal
                    
                    #detect if it's small blocks
                    if 500<=M["m00"] and M["m00"]<=1450:
                        blockarray=np.array([xyz_w[0],xyz_w[1],xyz_w[2],0,colornum,rad])
                        blockarray=blockarray.reshape(1, -1)
                        if self.BLOCKS is None:
                            self.BLOCKS=blockarray
                        else:
                            self.BLOCKS=np.append(self.BLOCKS,blockarray,axis=0)
                        
                        # self.BLOCKS[count]=np.array([xyz_w[0],xyz_w[1],xyz_w[2],0,colornum,rad])
                        # count=count+1

                    
                    #detect if it's big blocks
                    if 1500<=M["m00"] and M["m00"]<=4000:
                        blockarray=np.array([xyz_w[0],xyz_w[1],xyz_w[2],1,colornum,rad])
                        blockarray=blockarray.reshape(1, -1)
                        if self.BLOCKS is None:
                            self.BLOCKS=blockarray
                        else:
                            self.BLOCKS=np.append(self.BLOCKS,blockarray,axis=0)
                        # self.BLOCKS[count]=np.array([xyz_w[0],xyz_w[1],xyz_w[2],0,colornum,rad])
                        # count=count+1


        # cv2.waitKey(1)  # waits 1 ms
        


        # For each block detected...

        # LOCATIONS: world coordinates
        # SIZE: 0 = small, 1 = large
        # COLOR: 0 = red, 1 = orange, 2 = yellow, 3 = green, 4 = blue, 5 = violet
        # ROT: rotation angle of block (rad)

        #parameters = [LOCATIONS, SIZE, COLOR, ROT]
        # self.block_detections.append(parameters)

        blockA = [150, 125, 40, 1, 0, 0]
        blockB = [250, 125, 25, 0, 5, 0]
        blockC = [150, 25, 40, 1, 4, 0]
        blockD = [150, 225, 40, 1, 3, 0]
        blockE = [0,175,25,0,1,0]
        blockF = [100,225,25,0,0]
        
        self.block_detections = self.BLOCKS
        


    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        # self.grid_points  # nx2 for x,y
        intrinsicMat = np.array([[904.6,0,635.982],[0,905.29,353.06],[0,0,1]]) 
        # extrinsicMat = np.array([[1,0,0,0],[0,-0.9797,0,190],[0,0.2004,-0.9797,970],[0,0,0,1]])
        if self.world_coord_calib_flag:
            extrinsicMat = self.cam_extrinsic_maxtrix
        else:
            extrinsicMat = np.array([[1,0,0,0],[0,-0.9797,0,190],[0,0.2004,-0.9797,970],[0,0,0,1]])
        #print("self.grid_points shape: ", self.grid_points.shape)
        #print(self.grid_points_flattened.shape)
        # pdb.set_trace()

        newrowz =np.zeros((1,266))
        newrow_ones = np.ones((1,266))
        xyz_one_array = np.vstack([self.grid_points_flattened, newrowz])
        xyz_one_array = np.vstack([xyz_one_array, newrow_ones]) # 4x266
        Hprod = np.dot(extrinsicMat, xyz_one_array)  # 4x266
        depths_camera = Hprod[2,:]
        Imat = np.eye(3)
        column_to_be_added = np.array([[0], [0], [0]])
        Imat = np.hstack([Imat, column_to_be_added])
        P = (1/970) * np.dot(intrinsicMat, Imat)
        uv_mat =  np.dot(P, Hprod)
        # uv_mat = (1 / depths_camera) * np.dot(P, Hprod)  # use only first 2 rows of this to draw on opencv

        # points_xyz_c = np.dot(self.extrinsicMat, points_xyz_w)  # must be 4x20
        # projection_mat = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
        # projection_times_xyz_c = np.dot(projection_mat, points_xyz_c)
        # depths_camera = np.transpose(np.delete(points_xyz_c.transpose(), (0, 1,3), axis=1)) # stores only 3rd col
        # points_ones = np.ones(depths_camera.size)
        # points_uv = np.transpose((1 / depths_camera) * np.dot(self.intrinsicMat, projection_times_xyz_c))


        if not (self.cam_homography_matrix.size == 0):
            uv_mat = np.dot(self.cam_homography_matrix, uv_mat)

        modified_image = self.VideoFrame.copy() #NDArray[uint8]
        for idx in range(uv_mat.shape[1]):
            center_x = uv_mat[0,idx]
            center_y = uv_mat[1,idx]
            center_coords = (int(center_x), int(center_y))
            modified_image = cv2.circle(modified_image, center_coords, radius=5, color=(0,0,255), thickness=-1) # for the center

        self.GridFrame = modified_image


    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy() #NDArray[uint8]
        # Write your code here
        for tag in msg.detections:
            center_x = tag.centre.x
            center_y = tag.centre.y
            # print(type(center_x))
            # print("x: ",center_x, "y: ", center_y)
            # print(type(modified_image))
            center_coords = (int(center_x), int(center_y))
            self.tag_ids_centers[int(tag.id)] = center_coords
            self.tag_ids_centers_corners[int(tag.id)] = [center_coords]

            # if (tag.id <= 4):
            #     for corner in tag.corners:
            #         self.tag_ids_centers_corners[int(tag.id)].append((int(corner.x), int(corner.y)))

            # print(type(center_coords))
            modified_image = cv2.circle(modified_image, center_coords, radius=5, color=(0,0,255), thickness=-1) # for the center
            # print(type(modified_image))

            corners_list = tag.corners
            # 0 to 1
            modified_image = cv2.line(modified_image, (int(corners_list[0].x), int(corners_list[0].y)), (int(corners_list[1].x), int(corners_list[1].y)), (255,0,0), 3)
            modified_image = cv2.line(modified_image, (int(corners_list[1].x), int(corners_list[1].y)), (int(corners_list[2].x), int(corners_list[2].y)), (255,0,0), 3)
            modified_image = cv2.line(modified_image, (int(corners_list[2].x), int(corners_list[2].y)), (int(corners_list[3].x), int(corners_list[3].y)), (255,0,0), 3)
            modified_image = cv2.line(modified_image, (int(corners_list[3].x), int(corners_list[3].y)), (int(corners_list[0].x), int(corners_list[0].y)), (255,0,0), 3)

            modified_image = cv2.circle(modified_image, (int(corners_list[0].x), int(corners_list[0].y)), radius=5, color=(255,0,0), thickness=-1) #corner 0
            modified_image = cv2.circle(modified_image, (int(corners_list[1].x), int(corners_list[1].y)), radius=5, color=(0,255,0), thickness=-1) #corner 1
            modified_image = cv2.circle(modified_image, (int(corners_list[2].x), int(corners_list[2].y)), radius=5, color=(0,0,255), thickness=-1) #corner 2; corner 3 is not colored

            
            id_text = "ID: " + str(tag.id)
            modified_image = cv2.putText(modified_image, id_text, (int(corners_list[2].x) -70, int(corners_list[2].y)-70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA, False)


        self.TagImageFrame = modified_image

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)

        if not (self.camera.cam_homography_matrix.size == 0):
            # print("Applying homography correction to image")
            # print("cv_image.shape[1], cv_image.shape[0]: ", cv_image.shape[1], cv_image.shape[0]) # 1280 720
            cv_image = cv2.warpPerspective(cv_image, self.camera.cam_homography_matrix, (cv_image.shape[1], cv_image.shape[0]), flags=cv2.INTER_LINEAR)
            self.warped_img = cv_image
            self.modified_warped_img_flag =True
            # print("calibrated!!")
            # print("warpPerspective checkpoint")

        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("Block detections window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                self.camera.blockDetector()

                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
