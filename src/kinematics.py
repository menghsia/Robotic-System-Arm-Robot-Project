"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""
import math
import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    # Initialize the vectors
    length = []
    twist = []
    offset = []
    angle = []
    curr = []
    A = np.empty([4,4])
    T = np.eye(4)
    
    # Construct transformation matrix for link
    link = 5
    for i in range(link):
        curr = dh_params[i]

        angle = curr[0]
        offset = curr[1]
        length = curr[2]
        twist = curr[3]
        
        A = get_transform_from_dh(length, twist, offset, angle + joint_angles[i])

        # Get updated T matrix using transformation matrix at link A
        T = np.matmul(T,A)
    
    return T
    

def get_transform_from_dh(a, alpha, d, theta):

    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    length = a
    twist = alpha
    offset = d
    angle = theta

    A = [[math.cos(angle), -math.sin(angle) * math.cos(twist), math.sin(angle) * math.sin(twist), length * math.cos(angle)] , 
         [math.sin(angle), math.cos(angle) * math.cos(twist), -math.cos(angle) * math.sin(twist), length * math.sin(angle)] ,
         [0, math.sin(twist), math.cos(twist), offset] , 
         [0, 0, 0, 1]]

    return A


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    phi = math.atan2(T[1,0],-T[2,0])
    theta = math.acos(T[0,0])
    psi = math.atan2(T[0,1],T[0,2])
    
    return phi, theta, psi


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    x = T[0,3]
    y = T[1,3]
    z = T[2,3]

    phi, theta, psi = get_euler_angles_from_T(T)

    pose = [x,y,z,phi,theta,psi]

    return pose


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass

def euler2mat(phi,theta,psi):
    from scipy.spatial.transform import Rotation
    mat = Rotation.from_euler("zyz", (phi,theta,psi))
    return mat

def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    x = pose[0]
    y = pose[1]
    z = pose[2]
    phi = pose[3]
    theta = pose[4]
    psi = pose[5]

    angle = []
    offset = []
    length = []
    twist = []

    link = 5
    for i in range(link):
        curr = dh_params[i]
        print("Curr: ", curr)
        angle.append(curr[0])
        offset.append(curr[1])
        length.append(curr[2])
        twist.append(curr[3])

    joint_configs = []
    wrist = []

    if ((x**2 + y**2 + z**2) > 400):
        print("This pose is not reachable")
        return

    # Wrist Position
    wrist = np.transpose([x,y,z]) - 0.174 * euler2mat(phi, theta, psi) * np.transpose([0,0,1])
    xc = wrist[0]
    yc = wrist[1]
    zc = wrist[2]

    r2 = xc**2 + yc**2
    s2 = zc - offset[0]

    # Config One: Down, Up, Sum
    J11 = np.atan2(yc,xc)
    J13 = np.acos(((xc**2 + yc**2) - length[1]**2 - length[2]**2) / (2 * length[1] * length[2]))
    J12 = np.atan2(yc,xc) - np.atan2(length[2]*np.sin(J13), length[1] * length[2] * np.cos(J13))
    J14 = phi - (J13 - J12)
    configOne = [J11,J12,J13,J14]

    # Config Two: Up, Down, Sum
    J21 = np.atan2(yc,xc)
    J23 = -np.acos(((xc**2 + yc**2) - length[1]**2 - length[2]**2) / (2 * length[1] * length[2]))
    J22 = np.atan2(yc,xc) - np.atan2(length[2]*np.sin(J23), length[1] * length[2] * np.cos(J23))
    J24 = phi - (J23 - J22)
    configTwo = [J21,J22,J23,J24]

    # Config Three: Sum, Down, Up
    J31 = np.atan2(yc,xc)
    J33 = np.acos(((xc**2 + yc**2) - length[1]**2 - length[2]**2) / (2 * length[1] * length[2]))
    J34 = np.atan2(y,x) - np.atan2(length[2]*np.sin(J33), length[1] * length[2] * np.cos(J33))
    J32 = phi - (J33 - J34)
    configThree = [J31,J32,J33,J34]

    # Config Four: Sum, Up, Down
    J41 = np.atan2(yc,xc)
    J43 = -np.acos(((xc**2 + yc**2) - length[1]**2 - length[2]**2) / (2 * length[1] * length[2]))
    J44 = np.atan2(yc,xc) - np.atan2(length[2]*np.sin(J43), length[1] * length[2] * np.cos(J43))
    J42 = phi - (J23 - J22)
    configFour = [J41,J42,J43,J44]
   
    # Put the four configs into one array
    joint_configs[0] = configOne
    joint_configs[1] = configTwo
    joint_configs[2] = configThree
    joint_configs[3] = configFour

    return joint_configs

    