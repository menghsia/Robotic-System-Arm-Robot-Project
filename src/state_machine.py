"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from rxarm import RXArm, RXArmThread
from camera import Camera
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
        
        # for tags 1-4, including all corners X,Y,Z mm world coords, 20x4 !!!make depth 0
        self.apriltags_board_positions = np.array([[-250,-25, 0, 1],[250,-25, 0, 1],[250,275, 0, 1],[-250, 275,0, 1]])

        #self.intrinsicMat = np.array([[977.9586, 0, 629.698],[0, 968.400, 363.818],[0, 0, 1]]) # use this with new UVD calc, same as control station
        self.intrinsicMat = np.array([[904.6,0,635.982],[0,905.29,353.06],[0,0,1]]) #factory intrinsic matrix
        self.K_inv = np.linalg.inv(self.intrinsicMat)

        


        self.extrinsicMat = np.array([[1, 0, 0, 0],[0, -0.9797, -0.2004, 190],[0, 0.2004, -0.9797, 970],[0,0,0,1]])  #!!! signs are inconsistent with above use with new uvd calc
        self.extrinsicMat_inv = np.linalg.inv(self.extrinsicMat)

        points_xyz_w =  self.apriltags_board_positions.transpose() # must be 4x20
        points_xyz_c = np.dot(self.extrinsicMat, points_xyz_w)  # must be 4x20


        projection_mat = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
        projection_times_xyz_c = np.dot(projection_mat, points_xyz_c)
        # projection_times_xyz_c = projection_mat @ points_xyz_c


        depths_camera = np.transpose(np.delete(points_xyz_c.transpose(), (0, 1,3), axis=1)) # stores only 3rd col
        points_ones = np.ones(depths_camera.size)


        points_uv = np.transpose((1 / depths_camera) * np.dot(self.intrinsicMat, projection_times_xyz_c))

        # pdb.set_trace()
        # print("pointsuv shape: ", points_uv.shape)
        self.dest_points = points_uv[:, :2] #IN UV
        print("self.dest_points shape: ", self.dest_points.shape)
        # pdb.set_trace()

        # ___________
        # P = np.dot(self.intrinsicMat, self.extrinsicMat) # 4x4
        # # print("P: ", P)
        # uvd_mat = np.dot(P, self.apriltags_board_positions.transpose()) # 4xn
        # # print("uvd_mat: ", uvd_mat)

        # self.destpt1 = np.dot(P,self.apriltag1_position)
        # # dest_pt 1:  [ 381.32113472  590.65212668 1005.83505155    1.03092784]

        # self.dest_points = uvd_mat[:2, :]  #NOTE!!! source points are in UV
        # self.dest_points = self.dest_points.transpose()
        # # this is in xyz world coordinates (mm)
        # print("self dest pts SHAPE: ", self.dest_points.shape, "self dest pts: ", self.dest_points)  # 2x24 but we need 24x2
        

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

        if self.next_state == "clickImplementation":
            self.clickImplementation()

        if self.next_state == "eventOne":
            self.eventOne()

        if self.next_state == "eventTwo":
            self.eventTwo()

        if self.next_state == "eventThree":
            self.eventThree()

        if self.next_state == "eventFour":
            self.eventFour()

        if self.next_state == "Bonus":
            self.Bonus()


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

    def recover_homogenous_transform_pnp(self, image_points, world_points, K):
        '''
        Use SolvePnP to find the rigidbody transform representing the camera pose in
        world coordinates (not working)
        '''
        D = np.array([0.1615992933511734, -0.5091497302055359, -0.0018777191871777177,0.0004640672996174544, 0.45967552065849304])
        distCoeffs = D
        #print("world_points:",world_points)
        #print("image_points:",image_points)
        [_, R_exp, t] = cv2.solvePnP(world_points,
                                 image_points,
                                 K,
                                 distCoeffs,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(R_exp)
        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))


    def calibrate(self, camera_ids_tags):
        """!
        camera_ids_tags: includes center, 4 corners
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        # input1 = input("Enter calibration array: ")  

        apriltag_centers_corners_cv =[]
        pnp_points_uv=[]
        for tag_id, point_pair_list in camera_ids_tags.items():
            #print("TEST tag id order: ", tag_id)
            #print(type(point_pair_list))
            for point_pair in point_pair_list:
                if tag_id < 5:
                    apriltag_centers_corners_cv.append([point_pair[0], point_pair[1]])  # UV. To get depth, use the apriltags 3rd col
            for point_pair in point_pair_list:
                pnp_points_uv=np.append(pnp_points_uv, [point_pair[0], point_pair[1]])
            
            
        
        
        # NOTE update___________
        # K_inv = np.linalg.inv(intrinsicMat)  # use self.K_inv
        # ______________________

        src_pts = np.asanyarray(apriltag_centers_corners_cv) # in uv world coords in the original plane, 


        image = self.camera.VideoFrame.copy()
        margin = (image.shape[1] - 20/13*(image.shape[0]-20))/2
        dest_pts = np.array((0.25*(image.shape[1]-2*margin)+margin,10/13*(image.shape[0]-20)+10, 
                             0.75*(image.shape[1]-2*margin)+margin,10/13*(image.shape[0]-20)+10, 
                             0.75*(image.shape[1]-2*margin)+margin,4/13*(image.shape[0]-20)+10,0.25*(image.shape[1]-2*margin)+margin, 
                             4/13*(image.shape[0]-20)+10)).reshape(4,2)
        
        #############
        
        #print("pnp_points_uv:",pnp_points_uv)
        # pnp_points_uv=pnp_points_uv.reshape((6,2))
        pnp_points_uv=pnp_points_uv.reshape((4,2))
        # pnp_points_world=np.array([[-250,-25, 0],[250,-25, 0],[250,275, 0],[-250, 275,0],[0,175,150], [-350, 150, 100]])
        pnp_points_world=np.array([[-250,-25, 0],[250,-25, 0],[250,275, 0],[-250, 275,0]])
        
        depths_camera = np.transpose(np.delete(src_pts, (0, 1), axis=1))
        points_ones = np.ones(depths_camera.size)
        A_pnp = self.recover_homogenous_transform_pnp(pnp_points_uv.astype(np.float32), pnp_points_world.astype(np.float32), self.intrinsicMat)
        #print("A_pnp:",A_pnp)
        self.camera.cam_extrinsic_maxtrix = A_pnp
        #points_camera = np.transpose(depths_camera * np.dot(self.K_inv, np.transpose(np.column_stack((pnp_points_uv, points_ones)))))
        #points_transformed_pnp = np.dot(np.linalg.inv(A_pnp), np.transpose(np.column_stack((points_camera, points_ones))))

        ##########
        
        # pdb.set_trace()

        # print("calibrate func src_pts: ", src_pts)
        # print("dest_pts shape: ", dest_pts.shape)


        # self.homography_matrix = cv2.findHomography(src_pts, dest_pts)[0]
        self.camera.cam_homography_matrix = cv2.findHomography(src_pts, dest_pts)[0]
        # print(self.camera.cam_homography_matrix)
        
        self.camera.world_coord_calib_flag = True
        self.status_message = "Calibration - Completed Calibration"

        self.next_state = "idle"

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


    def clickImplementation(self):
        # Set state to clickImplementation
        self.current_state = "clickImplementation"

        GripperDownPose = [1.57,1.57,1.57]

        # Part 1: Pick up a block
        # Message to tell user to click on a block
        self.status_message = "Click on block to pickup"
        sys.stdout.flush()

        # Get click location from user

        while(self.camera.new_click == False):
            pass
      
        self.camera.new_click = False

        # Pickup Location
        x = round(self.camera.cs_x,2)
        y = round(self.camera.cs_y,2)
        z = round(self.camera.cs_z,2)
        phi = GripperDownPose[0]
        theta = GripperDownPose[1]
        psi = GripperDownPose[2]
        
        pose = [x, y, z, phi, theta, psi]
        self.pickup(pose)

        # Part 2: Drop off the block
        # Message to tell user to click on a drop location
        self.status_message = "Click location to drop block"
        sys.stdout.flush()


        # Get click location from user
        while(self.camera.new_click == False):
            pass
      
        self.camera.new_click = False

        # Dropoff Location
        x = self.camera.cs_x
        y = self.camera.cs_y
        z = self.camera.cs_z
        phi = GripperDownPose[0]
        theta = GripperDownPose[1]
        psi = GripperDownPose[2]

        pose = [x, y, z, phi, theta, psi]
        self.dropoff(pose)

        # Send arm to initial position
        self.setPosHome()
        
        # Set status back to idle
        self.next_state = "idle"


    # COMPETITION FUNCTIONS

    def eventOne(self):
        print("Event One")

        # Set status back to idle
        self.next_state = "idle"

    def eventTwo(self):
        print("Event Two")
        # Set status back to idle
        self.next_state = "idle"


    def eventThree(self):
        print("Event Three")

        # Set status back to idle
        self.next_state = "idle"


    def eventFour(self):
        print("Event Four")
        # Set status back to idle
        self.next_state = "idle"


    def Bonus(self):
        print("Bonus")

        # Set status back to idle
        self.next_state = "idle"



    # COMPETITION HELPER FUNCTIONS

    def setPosHome(self): 
        self.rxarm.set_positions([0.0,       0.0,      0.0,          0.0,        0.0])
        time.sleep(3)


    def pickup(self, pose):
        x = pose[0]
        y = pose[1]
        z = pose[2]
        phi = pose[3]
        theta = pose[4]
        psi = pose[5]

        # Add 100mm to z position so we don't smash into board
        z += 100
        pose = [x, y, z, phi, theta, psi]

        # Get needed angles from IK
        from kinematics import IK_geometric
        joint_configs = IK_geometric(self.rxarm.dh_params, pose) 

        # Move above desired position (+100mm in Z)
        self.rxarm.set_positions([round(joint_configs[1][0],1),       round(joint_configs[1][1],1),      round(joint_configs[1][2],1),          round(joint_configs[1][3],1),        round(joint_configs[1][0],1)])
        time.sleep(3)


        # Lower the gripper
        z += -100
        pose = [x, y, z, phi, theta, psi]

        # Get needed angles from IK
        joint_configs = IK_geometric(self.rxarm.dh_params, pose) 
    
        # Move to desired position
        self.rxarm.set_positions([round(joint_configs[1][0],1),       round(joint_configs[1][1],1),      round(joint_configs[1][2],1),          round(joint_configs[1][3],1),        round(joint_configs[1][0],1)])
        time.sleep(2)

        # Close the gripper
        self.rxarm.close_gripper()
        time.sleep(3)

            # Raise the gripper
        z += 100
        pose = [x, y, z, phi, theta, psi]

        # Get needed angles from IK
        joint_configs = IK_geometric(self.rxarm.dh_params, pose) 
    
        # Move to desired position
        self.rxarm.set_positions([round(joint_configs[1][0],1),       round(joint_configs[1][1],1),      round(joint_configs[1][2],1),          round(joint_configs[1][3],1),        round(joint_configs[1][0],1)])
        time.sleep(2)

    def dropoff(self, pose):
        x = pose[0]
        y = pose[1]
        z = pose[2]
        phi = pose[3]
        theta = pose[4]
        psi = pose[5]

        # Add 100mm to z position so we don't smash into board
        z += 100
        pose = [x, y, z, phi, theta, psi]

        # Get needed angles from IK
        from kinematics import IK_geometric
        joint_configs = IK_geometric(self.rxarm.dh_params, pose) 

        # Move above desired position (+100mm in Z)
        self.rxarm.set_positions([round(joint_configs[1][0],1),       round(joint_configs[1][1],1),      round(joint_configs[1][2],1),          round(joint_configs[1][3],1),        round(joint_configs[1][0],1)])
        time.sleep(3)


        # Lower the gripper
        z += -100
        pose = [x, y, z, phi, theta, psi]

        # Get needed angles from IK
        joint_configs = IK_geometric(self.rxarm.dh_params, pose) 
    
        # Move to desired position
        self.rxarm.set_positions([round(joint_configs[1][0],1),       round(joint_configs[1][1],1),      round(joint_configs[1][2],1),          round(joint_configs[1][3],1),        round(joint_configs[1][0],1)])
        time.sleep(2)

        # Open the gripper
        self.rxarm.open_gripper()
        time.sleep(3)

            # Raise the gripper
        z += 100
        pose = [x, y, z, phi, theta, psi]

        # Get needed angles from IK
        joint_configs = IK_geometric(self.rxarm.dh_params, pose) 
    
        # Move to desired position
        self.rxarm.set_positions([round(joint_configs[1][0],1),       round(joint_configs[1][1],1),      round(joint_configs[1][2],1),          round(joint_configs[1][3],1),        round(joint_configs[1][0],1)])
        time.sleep(2)

        

        


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
