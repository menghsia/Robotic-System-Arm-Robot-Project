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
        {'id': 'red', 'color': (10, 10, 127)},
        {'id': 'orange', 'color': (30, 75, 150)},
        {'id': 'yellow', 'color': (30, 150, 200)},
        {'id': 'green', 'color': (20, 60, 20)},
        {'id': 'blue', 'color': (100, 50, 0)},
        {'id': 'violet', 'color': (100, 40, 80)})
        )

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # world coordinates from control_station.py
        self.cs_x = None
        self.cs_y = None
        self.cs_z = None

        # for getting image after applying homography
        self.warped_img



    def retrieve_area_color(self, data, contour, labels):
        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(data, mask=mask)[:3]
        min_dist = (np.inf, None)
        for label in labels:
            d = np.linalg.norm(label["color"] - np.array(mean))
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
        
        if not (self.cam_homography_matrix.size == 0):        
            self.DepthFrameRGB = cv2.warpPerspective(self.DepthFrameRGB, self.cam_homography_matrix, (self.DepthFrameRGB.shape[1], self.DepthFrameRGB.shape[0]), flags=cv2.INTER_LINEAR)


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

        if not (self.cam_homography_matrix.size == 0):        
            self.DepthFrameRaw = cv2.warpPerspective(self.DepthFrameRaw, self.cam_homography_matrix, (self.DepthFrameRaw.shape[1], self.DepthFrameRaw.shape[0]), flags=cv2.INTER_LINEAR)


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

        if not (self.camera.cam_homography_matrix.size == 0):
            rgb_image = self.warped_img #NDArray[uint8]
            cnt_image = self.warped_img
        
        else:
            rgb_image = self.VideoFrame.copy() #NDArray[uint8]
            cnt_image = self.VideoFrame.copy()
        
        depth_data = self.DepthFrameRaw.copy()
        # depth_data = self.DepthFrameRGB.copy()
        # pdb.set_trace()
        # print("DEPTH ", depth_data)

        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.rectangle(mask, (98,14),(1190,703), 255, cv2.FILLED)  # board box
        cv2.rectangle(mask, (550,400),(720,720), 0, cv2.FILLED)  # arm box

        cv2.rectangle(cnt_image, (98,14),(1190,703), (255, 0, 0), 2)  # board box
        cv2.rectangle(cnt_image, (550,400),(720,720), (255, 0, 0), 2)  # arm box

        thresh = cv2.bitwise_and(cv2.inRange(depth_data, 500, 960), mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)

        for contour in contours:
            color = self.retrieve_area_color(rgb_image, contour, self.colors)
            theta = cv2.minAreaRect(contour)[2]
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                cx = 24
                cy = 24
            cv2.putText(cnt_image, color, (cx-30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
            cv2.putText(cnt_image, str(int(theta)), (cx, cy), self.font, 0.5, (255,255,255), thickness=2)
            #print(color, int(theta), cx, cy)

        self.BlockContourImg = cnt_image
        cv2.imshow("Block detections window", cv2.cvtColor(self.BlockContourImg, cv2.COLOR_RGB2BGR))



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
            cv2.namedWindow("Block detections window", cv2.WINDOW_NORMAL)
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
                    
                    cv2.imshow("Block detections window",
                        cv2.cvtColor(self.camera.BlockContourImg, cv2.COLOR_RGB2BGR))

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
