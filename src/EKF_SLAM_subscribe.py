import numpy as np
import pdb
import time
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import tf
import math

n_state = 3 # Number of state variables
n_landmarks = 10 # Number of landmarks

R = np.diag([0.01,0.01,0.01]) # Process noise covariance
Q = np.diag([0.00,0.05]) # sigma_r, sigma_phi

# ---> EKF Estimation Variables
mu = np.zeros((n_state+2*n_landmarks,1)) # State estimate (robot pose and landmark positions)
sigma = np.zeros((n_state+2*n_landmarks,n_state+2*n_landmarks)) # State uncertainty, covariance matrix

mu[:] = np.nan # Initialize state estimate with nan values
np.fill_diagonal(sigma,100) # Initialize state uncertainty with large variances, no correlations

Fx = np.block([[np.eye(3),np.zeros((n_state,2*n_landmarks))]]) # Used in both prediction and measurement updates

   
class Localization:
    def __init__(self, topic_name):
        self.prev_time = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_theta = 0
        self.sub = rospy.Subscriber(topic_name, Odometry, self.pose_callback, queue_size=4)

    def pose_callback(self,msg):
        """
        Callback function for the /pose topic.

        Parameters:
            msg (Path): Message containing the pose data.
        """
        global v, w

        current_time = msg.header.stamp.to_sec()
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

        euler = tf.transformations.euler_from_quaternion(quaternion)
        current_theta = euler[2]

        if self.prev_time is not None:
            # Calculate the difference in position
            dx = current_x - self.prev_x
            dy = current_y - self.prev_y
            dtheta = current_theta - self.prev_theta

            # Normalize dtheta to the range [-pi, pi]
            dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

            # Calculate the linear velocity (v)
            v = math.sqrt(dx**2 + dy**2)

            # Calculate the angular velocity (w)
            w = dtheta

        #rospy.loginfo("v: %f, w: %f", v, w)
        #rospy.loginfo("x: %f, y: %f. thetha : %f, time %f", current_x, current_y, current_theta,current_time)
        #rospy.loginfo("x: %f, y: %f. thetha : %f, time %f", self.prev_x, self.prev_y, self.prev_theta, self.prev_time)

        # Update the previous values
        self.prev_time = current_time
        self.prev_x = current_x
        self.prev_y = current_y    
        self.prev_theta = current_theta

class Arucos:
    def __init__(self, topic_name):
        self.sub = rospy.Subscriber(topic_name, Float32, self.measurements_callback, queue_size=4)

    def measurements_callback(msg):
        """
        Callback function for the /pose topic.

        Parameters:
            msg (Path): Message containing the pose data.
        """
        global measures

        # Extract x, y, and theta from the pose message
        measures = np.zeros(3) # List of measurements

        measures = msg.zs

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
    rx,py,theta = mu[0],mu[1],mu[2]
    v,w = u[0],u[1]
    Erro_r = np.diag([0,0,0])
    
    # Update state estimate mu with model
    state_model_mat = np.zeros((n_state,1)) # Initialize state update matrix from model
    state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w) if np.abs(w)>0.01 else v*np.cos(theta)# Update in the robot x position
    state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w) if np.abs(w)>0.01 else v*np.sin(theta) # Update in the robot y position
    state_model_mat[2] = w # Update for robot heading theta
    mu = mu + np.matmul(np.transpose(Fx),state_model_mat) # Update state estimate, simple use model with current state estimate
    
    # Update state uncertainty sigma
    state_jacobian = np.zeros((3,3)) # Initialize model jacobian
    state_jacobian[0,2] = (v/w)*np.cos(theta) - (v/w)*np.cos(theta+w) if w>0.01 else -v*np.sin(theta) # Jacobian element, how small changes in robot theta affect robot x
    state_jacobian[1,2] = (v/w)*np.sin(theta) - (v/w)*np.sin(theta+w) if w>0.01 else v*np.cos(theta)# Jacobian element, how small changes in robot theta affect robot y
    
    if (moving == 1):
        Erro_r = R
        
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
    rx,ry,theta = mu[0,0],mu[1,0],mu[2,0] # robot 
    delta_zs = [np.zeros((2,1)) for lidx in range(n_landmarks)] # A list of how far an actual measurement is from the estimate measurement
    Ks = [np.zeros((mu.shape[0],2)) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    Hs = [np.zeros((2,mu.shape[0])) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    
    for z in zs:
        (dist,phi,lidx) = z
        mu_landmark = mu[n_state+lidx*2:n_state+lidx*2+2] # Get the estimated position of the landmark
        if np.isnan(mu_landmark[0]): # If the landmark hasn't been observed before, then initialize (lx,ly)
            mu_landmark[0] = rx + dist*np.cos(phi+theta) # lx, x position of landmark
            mu_landmark[1] = ry+ dist*np.sin(phi+theta) # ly, y position of landmark
            mu[n_state+lidx*2:n_state+lidx*2+2] = mu_landmark # Save these values to the state estimate mu
        delta  = mu_landmark - np.array([[rx],[ry]]) # Helper variable
        q = np.linalg.norm(delta)**2 # Helper variable

        dist_est = np.sqrt(q) # Distance between robot estimate and and landmark estimate, i.e., distance estimate
        phi_est = np.arctan2(delta[1,0],delta[0,0])-theta; phi_est = np.arctan2(np.sin(phi_est),np.cos(phi_est)) # Estimated angled between robot heading and landmark
        z_est_arr = np.array([[dist_est],[phi_est]]) # Estimated observation, in numpy array
        z_act_arr = np.array([[dist],[phi]]) # Actual observation in numpy array
        delta_zs[lidx] = z_act_arr-z_est_arr # Difference between actual and estimated observation

        # Helper matrices in computing the measurement update
        Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
        Fxj[n_state:n_state+2,n_state+2*lidx:n_state+2*lidx+2] = np.eye(2)
        H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(q),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
                      [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
        H = H.dot(Fxj)
        Hs[lidx] = H # Added to list of matrices
        Ks[lidx] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q)) # Add to list of matrices
        
    # After storing appropriate matrices, perform measurement update of mu and sigma
    mu_offset = np.zeros(mu.shape) # Offset to be added to state estimate
    sigma_factor = np.eye(sigma.shape[0]) # Factor to multiply state uncertainty
    
    for lidx in range(n_landmarks):
        mu_offset += Ks[lidx].dot(delta_zs[lidx]) # Compute full mu offset
        sigma_factor -= Ks[lidx].dot(Hs[lidx]) # Compute full sigma factor
        
    mu = mu + mu_offset # Update state estimate
    sigma = sigma_factor.dot(sigma) # Update state uncertainty
    
    return mu,sigma

if __name__ == '__main__':

    # Initialize state variables

    x_init = np.array([0,0,np.pi/2]) # px, py, theta
    mu[0:3] = np.expand_dims(x_init,axis=1)
    sigma[0:3,0:3] = 0.1*np.eye(3)
    sigma[2,2] = 0 
    moving = 0
    u = np.array([0.,0.]) # Controls: u[0] = forward velocity, u[1] = angular velocity

    rospy.init_node('localization_node')
    rospy.init_node('Arucos_node')

    subscriber_1 = Localization("/odom")
    subscriber_2 = Arucos("/Arucos_topic")

    u[0], u[1] = v , w
    zs = measures
    
    mu, sigma = prediction_update(mu,sigma,u,moving)
    mu, sigma = measurement_update(mu,sigma,zs)

    rospy.spin()