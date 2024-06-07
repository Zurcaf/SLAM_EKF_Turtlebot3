#! /usr/bin/env python3

import numpy as np
import pdb
import time
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import tf
import math
import cv2
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Float32, Header
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize empty data lists
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')

n_state = 3 # Number of state variables
n_landmarks = 50 # Number of landmarks

R = np.diag([0.01,0.01,0.1]) # Process noise covariance
Q = np.diag([0.00001,0.00001]) # sigma_r, sigma_phi

# ---> EKF Estimation Variables
mu = np.zeros((n_state+2*n_landmarks,1)) # State estimate (robot pose and landmark positions)
sigma = np.zeros((n_state+2*n_landmarks,n_state+2*n_landmarks)) # State uncertainty, covariance matrix

mu[:] = np.nan # Initialize state estimate with nan values
np.fill_diagonal(sigma,100) # Initialize state uncertainty with large variances, no correlations

Fx = np.block([[np.eye(3),np.zeros((n_state,2*n_landmarks))]]) # Used in both prediction and measurement updates

# Define camera parameters as global variables
CAMERA_MATRIX = np.array([[490.100238, 0.000000, 325.559625],
                          [0.000000, 489.477285, 240.754324],
                          [0.000000, 0.000000, 1.000000]], dtype=np.float32)

DIST_COEFFS = np.array([0.122115, -0.212824, 0.000813, 0.002590, 0.000000], dtype=np.float32)

# Define marker size as a global variable
MARKER_SIZE = 0.15  # Replace with your actual marker size


class Moving:
    def __init__(self):
        
        self.moving=0
        self.sub = rospy.Subscriber("/cmd_vel_rc100", Twist, self.moving_callback, queue_size=1)

    def moving_callback(self,msg):
        """
        Callback function for the /pose topic.

        Parameters:
            msg (Path): Message containing the pose data.
        """
        
        current_x = round(msg.linear.x,2)
        current_y = round(msg.linear.y,2)
        current_z = round(msg.angular.z,2)

        if( (current_x !=0) or (current_y !=0) or (current_z !=0) ):
            self.moving = 1
        else: 
            self.moving = 0
        
    def get_data(self):
        return self.moving
    
class Localization:
    def __init__(self, topic_name):
        self.prev_time = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_theta = 0
        self.mu = mu
        self.v = 0
        self.w = 0
        self.current_theta = 0
        self.sub = rospy.Subscriber(topic_name, Odometry, self.pose_callback, queue_size=1)

    def pose_callback(self,msg):
        """
        Callback function for the /pose topic.

        Parameters:
            msg (Path): Message containing the pose data.
        """
        rate = rospy.Rate(8)
        current_x = round(msg.pose.pose.position.x,2)
        current_y = round(msg.pose.pose.position.y,2)

        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.current_theta = round(euler[2],4)
        # print("current_theta:",self.current_theta)
        
        # Calculate the difference in position
        dx = current_x - self.prev_x
        dy = current_y - self.prev_y
        dtheta = self.current_theta - self.prev_theta

        # Normalize dtheta to the range [-pi, pi]
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

        # Calculate the linear velocity (v)
        self.v = math.sqrt(dx**2 + dy**2)

        # Calculate the angular velocity (w)
        self.w = dtheta

        #print(self.v,self.w)
        
        # Update the previous values
        self.prev_x = current_x
        self.prev_y = current_y 
        self.prev_theta = self.current_theta

        dist_x = round(msg.pose.pose.position.x,2)
        dist_y = round(msg.pose.pose.position.y,2)
        dist = round(math.sqrt(dist_x**2 + dist_y**2),2)
        
        
        # print("dist:", dist)
        # print("theta:",current_theta)
        rate.sleep()

    def get_data(self):
        return self.v , self.w , self.current_theta 


class ArucoDetector:
    def __init__(self,):
        
        # Create a CvBridge object
        self.bridge = CvBridge() 
        self.measures = [0,0,0]
        # Initialize the ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()

        # Subscribe to the input compressed image topic
        self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_callback)

    def image_callback(self, msg):

        try:
            # Convert the ROS CompressedImage message to an OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Extract the header from the compressed image message
        header = msg.header
        timestamp = header.stamp.to_sec()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        #print_mu(self.mu)
        
        # If markers are detected
        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)

            for i in range(len(ids)):
                # Draw the detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Calculate distance and bearing
                tvec = tvecs[i][0]
                distance = np.linalg.norm(tvec)
                bearing = np.arctan2(tvec[0], tvec[2])

                # Extract x, y, and theta from the pose message
                self.measures = np.zeros(3) # List of measurements
                
                self.measures = np.array([distance, bearing, int(ids[i][0])])

                # Draw axis if the function is available
                if hasattr(cv2.aruco, 'drawAxis'):
                    cv2.aruco.drawAxis(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], 0.1)
                else:
                    # Custom implementation of drawAxis if not available
                    self.custom_draw_axis(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], 0.1)
        else :
            self.measures = [0,0,0]
    
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        
        # Optionally display the frame
    def custom_draw_axis(self, image, camera_matrix, dist_coeffs, rvec, tvec, length):
        """ Draw custom axis on the image """
        points = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]).reshape(-1, 3)
        axis_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
        axis_points = axis_points.reshape(-1, 2)
        origin = tuple(axis_points[3].ravel().astype(int))
        image = cv2.line(image, origin, tuple(axis_points[0].ravel().astype(int)), (0, 0, 255), 3)
        image = cv2.line(image, origin, tuple(axis_points[1].ravel().astype(int)), (0, 255, 0), 3)
        image = cv2.line(image, origin, tuple(axis_points[2].ravel().astype(int)), (255, 0, 0), 3) 

    def get_data(self):
        return self.measures


def prediction_update(mu,sigma,u,moving):
    '''
    This function performs the prediction step of the EKF. Using the linearized motion model, it
    updates both the state estimate mu and the state uncertainty sigma based on the model and known
    control inputs to the robot.
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - u: model input
     - dt: discretization time of continuous model
    Outpus:
     - mu: updated state estimate
     - sigma: updated state uncertainty
    '''
    rx,ry,theta = mu[0],mu[1],mu[2]
    v,w = u[0],u[1]
    Erro_r = R
    
    if(moving == 0):
        Erro_r =  np.diag([0,0,0])
    # Update state estimate mu with model
    
    state_model_mat = np.zeros((n_state,1)) # Initialize state update matrix from model
    state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w) if np.abs(w)>0.01 else v*np.cos(theta)# Update in the robot x position
    state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w) if np.abs(w)>0.01 else v*np.sin(theta) # Update in the robot y position
    state_model_mat[2] = 0 # Update for robot heading theta
    mu = mu + np.matmul(np.transpose(Fx),state_model_mat) # Update state estimate, simple use model with current state estimate
   
    
    # Update state uncertainty sigma
    state_jacobian = np.zeros((3,3)) # Initialize model jacobian
    state_jacobian[0,2] = (v/w)*np.cos(theta) - (v/w)*np.cos(theta+w) if np.abs(w)>0.01 else -v*np.sin(theta) # Jacobian element, how small changes in robot theta affect robot x
    state_jacobian[1,2] = (v/w)*np.sin(theta) - (v/w)*np.sin(theta+w) if np.abs(w)>0.01 else v*np.cos(theta)# Jacobian element, how small changes in robot theta affect robot y   
        
    G = np.eye(sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian).dot(Fx) # How the model transforms uncertainty
    sigma = G.dot(sigma).dot(np.transpose(G)) + np.transpose(Fx).dot(Erro_r).dot(Fx) # Combine model effects and stochastic noise


    return mu,sigma


def measurement_update(mu,sigma,zs):
    '''
    This function performs the measurement step of the EKF. Using the linearized observation model, it
    updates both the state estimate mu and the state uncertainty sigma based on range and bearing measurements
    that are made between robot and landmarks.
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - zs: list of 3-tuples, (dist,phi,lidx) from measurement function
    Outpus:
     - mu: updated state estimate
     - sigma: updated state uncertainty
    '''
    rx,ry,theta = mu[0,0],mu[1,0],mu[2,0] # Unpack robot pose
    delta_zs = [np.zeros((2,1)) for lidx in range(n_landmarks)] # A list of how far an actual measurement is from the estimate measurement
    Ks = [np.zeros((mu.shape[0],2)) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    Hs = [np.zeros((2,mu.shape[0])) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
     
    dist = zs[0] # Unpack the measurement
    phi = zs[1]
    lidx = zs[2] # Unpack the landmark index
    lidx = int(lidx)

    # print("measures",dist,phi,lidx)
    # print("mu prior",rx,ry,theta)

    Q_matrix = Q
    if( lidx > 50 or dist > 4):
        return mu, sigma
    
    mu_landmark = mu[n_state+lidx*2:n_state+lidx*2+2] # Get the estimated position of the landmark
    
    if((mu_landmark[0] >  300)   or   (mu_landmark[0] <  -300 )):
        return mu, sigma
    
    if np.isnan(mu_landmark[0]): # If the landmark hasn't been observed before, then initialize (lx,ly)
        mu_landmark[0] = rx + dist*np.cos(phi+theta) # lx, x position of landmark
        mu_landmark[1] = ry+ dist*np.sin(phi+theta) # ly, y position of landmark
        mu[n_state+lidx*2:n_state+lidx*2+2] = mu_landmark # Save these values to the state estimate mu
    delta  = mu_landmark - np.array([[rx],[ry]]) # Helper variable
    q = np.linalg.norm(delta)**2 # Helper variable


    dist_est = np.sqrt(q) # Distance between robot estimate and and landmark estimate, i.e., distance estimate
    phi_est = np.arctan2(delta[1,0],delta[0,0])-theta; phi_est = np.arctan2(np.sin(phi_est),np.cos(phi_est)) # Estimated angled between robot heading and landmark

    z_est_arr = np.array([[dist_est],[phi_est]]) # Estimated observation, in numpy array
    # print("z_est_arr",z_est_arr)
    z_act_arr = np.array([[dist],[phi]]) # Actual observation in numpy array
    # print("z_act_arr",z_act_arr)


    delta_zs[lidx] = z_act_arr-z_est_arr # Difference between actual and estimated observation
    # print("delta_zs[lidx] after update", delta_zs[lidx] )

    # Helper matrices in computing the measurement update
    Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
    Fxj[n_state:n_state+2,n_state+2*lidx:n_state+2*lidx+2] = np.eye(2)
    H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(q),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
                    [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
    H = H.dot(Fxj)
    Hs[lidx] = H # Added to list of matrices
    Ks[lidx] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q_matrix)) # Add to list of matrices
    
    # After storing appropriate matrices, perform measurement update of mu and sigma
    mu_offset = np.zeros(mu.shape) # Offset to be added to state estimate
    sigma_factor = np.eye(sigma.shape[0]) # Factor to multiply state uncertainty
   
    for lid in range(n_landmarks):
        mu_offset += Ks[lid].dot(delta_zs[lid]) # Compute full mu offset
        sigma_factor -= Ks[lid].dot(Hs[lid]) # Compute full sigma factor
        # if(mu_offset[lid] != 0):
        #     print("mu_offset[lid]",lid,mu_offset[lid])
        #     print("delta_zs",lid,delta_zs[lid])
       
   
    mu = mu + mu_offset # Update state estimate
    sigma = sigma_factor.dot(sigma) # Update state uncertainty


    return mu,sigma


def print_mu(mu):
    lidx=0
    for lidx in range(51):
    # Calculate the starting and ending index
        start_idx = n_state + lidx * 2
        end_idx = start_idx + 2
        
        # Ensure we don't go out of bounds
        if end_idx > len(mu):
            break

        # Get the current element (an array of 2 values, x and y)
        elements = mu[start_idx:end_idx]
        

        x, y = elements

        if (math.isnan(x) == False):
            print(lidx,np.round(x,2),np.round(y,2))

def init():
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.grid(True)

    return ln

def update(mu):

    elements_plot = np.zeros((51,3))

    lidx=0

    for lidx in range(51):
    # Calculate the starting and ending index
        start_idx = n_state + lidx * 2
        end_idx = start_idx + 2
        
        # Ensure we don't go out of bounds
        if end_idx > len(mu):
            break

        # Get the current element (an array of 2 values, x and y)
        elements = mu[start_idx:end_idx]
        

        x, y = elements

        if (x != 0):

            elements_plot[lidx][0] = lidx
            elements_plot[lidx][1] = x
            elements_plot[lidx][2] = y

    # print(elements_plot)
    # print ("")

    ln.set_data(elements_plot[:,1], elements_plot[:,2])
    ax.relim()
    ax.autoscale_view()

if __name__ == '__main__':

    # Initialize state variables

    x_init = np.array([0,0,0]) # px, py, theta
    mu[0:3] = np.expand_dims(x_init,axis=1)
    sigma[0:3,0:3] = 0.1*np.eye(3)
    v=0 
    w=0
    sigma[2,2] = 0 
    rospy.init_node('ekf_slam')

    u = np.array([0.,0.]) # Controls: u[0] = forward velocity, u[1] = angular velocity
    rate = rospy.Rate(10)
    mu[2] = 0


    # Initialize the plot
    init()

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Landmark Position Estimation')

    # Display the plot
    plt.ion()  # Turn on interactive mode
    plt.show()

    try:

        subscriber_3 = Moving()
        subscribe_1 = ArucoDetector() 
        subscriber_2 = Localization("/odom")
         
        while not rospy.is_shutdown():

            measures = subscribe_1.get_data()
            v , w , theta= subscriber_2.get_data() 
            moving = subscriber_3.get_data()
            mu[2] = theta
            u[0], u[1] = v , w
            mu, sigma = prediction_update(mu,sigma,u,moving)
            # print("measures:",measures)
            # print("mu",mu[0],mu[1],mu[2])
            
            if( measures[0] != 0):
                # print("measures:",measures[1])
                
                zs = measures
                mu, sigma = measurement_update(mu,sigma,zs)

            # print_mu(mu)
            # print("theta",mu[2])

            update(mu)
            plt.pause(0.1)  # Pause for a short interval to allow for plot update
            
            rate.sleep()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Get Interrupt Message")  
        
    plt.ioff()
    plt.show()

    rospy.loginfo("Quitting...")
        
    
