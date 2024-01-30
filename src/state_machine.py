"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from rxarm import RXArm, RXArmThread
import sys
import cv2
import pdb

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]
        
        self.recorded_waypoints = []
        self.recorded_gripper_position = []
        self.gripper_open_flag = True
        self.world_coord_calib_flag = False

        # self.apriltags_board_positions = np.array([[-250,-25, -1049, 1],[250,-25, -1043, 1],[250,275, -988, 1],[-250,275,-995, 1]])  #4x2 X Y Z 1 with base link origin
        # self.apriltags_board_positions = np.array([[-250,-200, -1049, 1],[250,-200, -1043, 1],[250,100, -988, 1],[-250,100,-995, 1]])  #4x2 X Y Z 1 with center board origin
        # self.apriltags_board_positions = np.array([[-150,100, -1049, 1],[350,100, -1043, 1],[350,400, -988, 1],[-150,400,-995, 1]])  #4x2 with bottom left board origin

        # order matters
        # for tags 1-8, corners included only for 1-4
        # self.apriltags_board_positions = np.array([[-150,100, -1050, 1], [-165, 85, -1053, 1], [-135, 85, -1052,1], [-135, 115, -1048, 1], [-165, 115, -1046,1],
        #                                            [350,100, -1044, 1], [335, 85, -1047, 1], [365, 85, -1048, 1], [365, 115, -1041,1], [335, 115, -1040,1],
        #                                            [350,400, -990, 1], [335, 385, -992,1], [365, 385, -991,1], [365, 415, -986,1], [335, 415, -986,1],
        #                                            [-150,400,-996, 1], [-165, 385, -998,1], [-135, 385, -998,1], [-135, 415, -992,1], [-165, 415, -993,1],
        #                                            [100,450,-961,1], [-250,250, -999,1], [0,0,-1041,1], [200,200,-1003,1]])
        
        # for tags 1-4, including all corners X,Y,Z mm world coords
        self.apriltags_board_positions = np.array([[-150,100, -1050, 1], [-165, 85, -1053, 1], [-135, 85, -1052,1], [-135, 115, -1048, 1], [-165, 115, -1046,1],
                                            [350,100, -1044, 1], [335, 85, -1047, 1], [365, 85, -1048, 1], [365, 115, -1041,1], [335, 115, -1040,1],
                                            [350,400, -990, 1], [335, 385, -992,1], [365, 385, -991,1], [365, 415, -986,1], [335, 415, -986,1],
                                            [-150,400,-996, 1], [-165, 385, -998,1], [-135, 385, -998,1], [-135, 415, -992,1], [-165, 415, -993,1]])

        self.apriltag1_position = np.array([-250,-200, -1049, 1])

        self.intrinsicMat = np.array([[977.9586,0,629.698, 0],[0,968.400,363.818, 0],[0,0,1000, 0], [0,0,0,1000]]) / 970
        # self.intrinsicMat = np.array([[977.9586,0,629.698],[0,968.400,363.818],[0,0,1]]) # use this
        self.K_inv = np.linalg.inv(self.intrinsicMat)
        # intrinsicMat = np.array([[904.6,0,635.982, 0],[0,905.29,353.06, 0],[0,0,1000, 0], [0,0,0,1000]]) / 970 
        self.extrinsicMat = np.array([[1,0,0,0],[0,-0.9797,-0.2004,0.19],[0,0.2004,-0.9797,.970],[0,0,0,1]])

        # self.extrinsicMat = np.array([[1,0,0,0],[0,0.9797,-0.2004,190],[0,0.2004,0.9797,970],[0,0,0,1]])  # use with new uvd calc

        # points_xy = np.delete(self.apriltags_board_positions, -1, axis=1)  # stores only 1st and 2nd cols of apriltag board pos
        # depths_camera = np.transpose(np.delete(self.apriltags_board_positions, (0, 1), axis=1)) # stores only 3rd col of points_uvd
        # points_ones = np.ones(depths_camera.size)

        # # pdb.set_trace()

        # points_uv = np.transpose((1 / depths_camera) * np.dot(self.intrinsicMat,np.transpose(np.column_stack((points_xy, points_ones)))))
        # print("pointsuv shape: ", points_uv.shape)
        # self.dest_points = points_uv[:2, :] #IN UV

        # ___________
        P = np.dot(self.intrinsicMat, self.extrinsicMat) # 4x4
        # print("P: ", P)
        uvd_mat = np.dot(P, self.apriltags_board_positions.transpose()) # 4xn

        # print("uvd_mat: ", uvd_mat)

        self.destpt1 = np.dot(P,self.apriltag1_position)
        # dest_pt 1:  [ 381.32113472  590.65212668 1005.83505155    1.03092784]

        self.dest_points = uvd_mat[:2, :]  #NOTE!!! source points are in UVD
        self.dest_points = self.dest_points.transpose()
        #  # this is in xyz world coordinates (mm)
        print("self dest pts SHAPE: ", self.dest_points.shape, "self dest pts: ", self.dest_points)  # 2x24 but we need 24x2
        
        # self.dest_points = self.apriltags_board_positions[:, :2]
        # print("dest_pt 1: ", self.destpt1)
        self.homography_matrix = []  #initialized empty


    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "record_waypoint":
            self.record_waypoint()

        if self.next_state == "execute_waypoints":
            self.execute_waypoints()



    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        for waypoint in self.waypoints:
            self.rxarm.set_positions(waypoint)
            time.sleep(4)
            
        self.next_state = "idle"

    def execute_waypoints(self):
        """!
        @brief      Go through all recorded waypoints
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute_waypoints"

        for idx, waypoint in enumerate(self.recorded_waypoints):
            print("idx: ", idx)
            self.rxarm.set_positions(waypoint)
            time.sleep(1.5)
            
            if self.recorded_gripper_position[idx] is True:
                self.rxarm.gripper.release()
            else:
                self.rxarm.gripper.grasp()

            time.sleep(2)
            
        self.next_state = "idle"
   
    # def close_gripper(self):
    #     print("in close gripper")
    #     self.status_message = "closing gripper"
    #     self.rxarm.gripper.grasp()
    #     self.gripper_open_flag = False

    # def open_gripper(self):
    #     print("in open gripper")
    #     self.status_message = "releasing gripper"
    #     self.rxarm.gripper.release()
    #     self.gripper_open_flag = True

    def record_waypoint(self):
        print("entered record button function")
        sys.stdout.flush()

        self.current_state = "record_waypoint"
        curr_position = self.rxarm.get_positions()
        print(curr_position)
        self.recorded_waypoints.append(curr_position)

        # record gripper open/close
        self.recorded_gripper_position.append(self.rxarm.gripper_open_flag)
        print("self.recorded_gripper_position: ", self.recorded_gripper_position)

        self.status_message = "recorded waypoint"

        print(self.recorded_waypoints)
        sys.stdout.flush()
        self.next_state = "idle"

    def calibrate(self, camera_ids_tags):
        """!
        camera_ids_tags: includes center, 4 corners
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        # input1 = input("Enter calibration array: ")  
        print(camera_ids_tags) 
        apriltag_centers_corners_cv =[]
        for tag_id, point_pair_list in camera_ids_tags.items():
            print("TEST tag id order: ", tag_id)
            print(type(point_pair_list))
            for point_pair in point_pair_list:
                apriltag_centers_corners_cv.append([point_pair[0], point_pair[1]])  # UV. To get depth, use the apriltags 3rd col

        # NOTE update___________
        # K_inv = np.linalg.inv(intrinsicMat)
        # ______________________

        src_pts = np.asanyarray(apriltag_centers_corners_cv) # in uvd world coords in the original plane, 
        # print("apriltag_centers_corners_cv: ", apriltag_centers_corners_cv)
        print("shape of src points: ", src_pts.shape, " src_pts: ", src_pts)

        dest_pts = self.dest_points # 24x2 in XYZ
        
        
        # print("calibrate func src_pts: ", src_pts)
        print("calibrate func dest_pts: ", dest_pts)

        # self.homography_matrix = cv2.findHomography(src_pts, dest_pts)[0]
        self.camera.cam_homography_matrix = cv2.findHomography(src_pts, dest_pts)[0]
        print(type(self.camera.cam_homography_matrix))
        
        self.world_coord_calib_flag = True
        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        # print("debug 1")
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)