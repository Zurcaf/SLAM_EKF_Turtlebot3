import numpy as np
import math

from copy import deepcopy
from ipywidgets import *
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import Odometry
from fiducial_msgs.msg import FiducialTransformArray
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion

NUM_PARTICLES = 100 
NUM_LANDMARKS = 200 

DT = 1 #deta t

R = np.diag([0.05, np.deg2rad(4)])**2 #!odom
Q = np.diag([0.05, np.deg2rad(3)])**2  #!aruco



class Particle:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.first = True
        self.weight = 1/NUM_PARTICLES

        self.landPos = np.zeros((NUM_LANDMARKS, 2)) # 2 pq é 2D
        self.landCov = np.zeros((NUM_LANDMARKS*2, 2)) # 2 pq é 2D

class OdomSubscriber:
    def __init__(self, topic_name, particles, u):
        self.sub = rospy.Subscriber(topic_name, Odometry, self.Callback)
        self.last_pos = np.array([0, 0, 0])
        self.raw_pos = np.array([0, 0, 0])

        self.raw_pos = self.raw_pos.astype(np.float64)
        self.last_pos = self.last_pos.astype(np.float64)

        self.particles = particles
        self.u = u
    
    def Callback(self, msg):
        global odom_hist
        self.raw_pos[0] = msg.pose.pose.position.x
        self.raw_pos[1] = msg.pose.pose.position.y
        
        raw_ori_quat = msg.pose.pose.orientation
        self.raw_pos[2] = euler_from_quaternion((raw_ori_quat.x, raw_ori_quat.y, raw_ori_quat.z, raw_ori_quat.w))[2] 
        pos = (self.raw_pos - self.last_pos)

        if(not (np.any(abs(pos)==0))):
            odom_hist.append(self.raw_pos.copy())
        if(np.any(abs(pos) >= 0.0000001)): 
            self.last_pos = self.raw_pos.copy()            

            distance = np.sqrt(pos[0]**2 + pos[1]**2)
            self.u = np.array([[distance, pos[2]]]).reshape(2,1)
            for particle in self.particles:
                if(particle.first):
                    particle.first = False
                    particle.x = self.raw_pos[0]
                    particle.y = self.raw_pos[1]
                    particle.yaw = self.raw_pos[2]
                    continue
                sample_pose(particle, self.u)

class ArucoSubscriber:
    def __init__(self, topic_name, particles, u):
        self.sub = rospy.Subscriber(topic_name, FiducialTransformArray, self.Callback)
        self.particles = particles
        self.u = u
    
    def Callback(self, msg):
        #print(msg)
        global history
        global odom_hist

        global initial_time
        global total
        global missed
        total += 1
        print("Percentage missed", (missed/total)*100)
        if initial_time is None:
            initial_time = rospy.Time.now().to_sec() - msg.header.stamp.to_sec()
        
        time = rospy.Time.now().to_sec() - initial_time
        # If the aruco recording was to long ago, ignore it
        if time - msg.header.stamp.to_sec() > 0.5:
            missed += 1
            return

        z = -np.ones((3, NUM_LANDMARKS))
        for i in range(NUM_LANDMARKS):
            try:
                y = float(msg.transforms[i].transform.translation.x)
                x = float(msg.transforms[i].transform.translation.z)
                id = int(msg.transforms[i].fiducial_id)
            except:
                break
            #filter some ids

            # #if id is odd
            # if(id%2==1):
            #     continue
            dist = np.sqrt(x**2 + y**2)
            angle = norm_angle(math.atan2(-y,x))
            zi = np.array([dist, norm_angle(angle), id]).reshape(3, 1) 
            z[:, id] = zi.squeeze()


        self.particles = update_with_observation(self.particles, z)


        self.particles = resampling(self.particles)
        history.append(deepcopy(self.particles))

        #!toplot best particle
        pltparticles = []
        plt.clf()
        for i in range(len(history)):
            if(i==0): 
                continue
            maxwj = 0
            for j in range(len(history[i])):
                if(self.particles[j].weight > self.particles[maxwj].weight):
                    maxwj = j
            pltparticles.append(deepcopy(history[i][maxwj]))   
        if(len(pltparticles) > 0):
            pub = rospy.Publisher('/toplot', Float64MultiArray, queue_size=10)
            data_to_send = Float64MultiArray()  # the data to be sent, initialise the array
            data_to_send.data = [] # assign the array with the value you want to send
            data_to_send.data.append(pltparticles[-1].x)
            data_to_send.data.append(pltparticles[-1].y)
            for i in range(NUM_LANDMARKS):
                if pltparticles[-1].landPos[i, 0] != 0:
                    data_to_send.data.append(pltparticles[-1].landPos[i, 0])
                    data_to_send.data.append(pltparticles[-1].landPos[i, 1])
                    data_to_send.data.append(pltparticles[-1].landCov[i, 0])
                    data_to_send.data.append(pltparticles[-1].landCov[i, 1])
                    data_to_send.data.append(i)
            pub.publish(data_to_send)
            data_to_send.data.clear()
        return



def sample_pose(particle, u):
    part = np.zeros((3,1)) #x,y,yaw
    part[0] = particle.x
    part[1] = particle.y
    part[2] = particle.yaw
    #sampling
    nois_u = u + (np.random.randn(1,2) @ R).T #adding noise ao input(odom) 
    part = motion_model(part, nois_u) 
    #part = motion_model(part, u) # to see without noise
    #post sampling
    particle.x = part[0,0]
    particle.y = part[1,0]
    particle.yaw = part[2,0]

    return particle

def motion_model(x, u):
    F = np.array([[1,0,0],
                [0,1,0],
                [0,0,1]])


    B = np.array([[DT*math.cos(x[2,0]), 0],
                [DT*math.sin(x[2,0]), 0],
                [0,DT]])
                
    #x(t+1) = x + cos(yaw)* dx
    #y(t+1) = y + sin(yaw)* dx
    #yaw(t+1) = yaw + dtheta

    x = F@x + B@u

    x[2, 0] = norm_angle(x[2, 0])

    return x

def norm_angle(angle): 
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def update_with_observation(particles, z):
    for j in range(len(z[0, :])):
        lm_id = int(z[2, j])


        if(lm_id!=-1):
            for particle in particles:
                # Check if landmark ID is already known
                if np.any(particle.landPos[lm_id] != np.zeros((2, 1))): #already known
                    w = compute_weight(particle, z[:,j], Q)
                    particle.weight *= w
                    
                    particle = update_landmark(particle, z[:,j], Q)
                else: #new
                    particle = add_new_lm(particle, z[:,j], Q)
    return particles

def compute_weight(particle, z, Q):
    lm_id = int(z[2]) 
    xf = np.array(particle.landPos[lm_id, :2]).reshape(2, 1)
    Pf = np.array(particle.landCov[2 * lm_id:2 * lm_id + 2, :])

    zp, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = norm_angle(dx[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print("singular")
        return 1.0

    num = math.exp(-0.5 * dx.T @ invS @ dx)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))
    w = num / den

    return w

def add_new_lm(particle, z, Q):
    r = z[0]
    b = z[1]
    lm_id = int(z[2])


    s = math.sin(norm_angle(particle.yaw + b))
    c = math.cos(norm_angle(particle.yaw + b))


    particle.landPos[lm_id, 0] = particle.x + r * c
    particle.landPos[lm_id, 1] = particle.y + r * s
    
    Gz = np.array([[c, -r * s],
                   [s, r * c]])

    particle.landCov[2 * lm_id:2 * lm_id + 2, :] = Gz @ Q @ Gz.T

    return particle

def update_landmark(particle, z, Q):
    lm_id = int(z[2])

    xf = np.array(particle.landPos[lm_id, :2]).reshape(2, 1)
    Pf = np.array(particle.landCov[2 * lm_id:2 * lm_id + 2, :])

    zp, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = norm_angle(dz[1, 0])

    xf, Pf = update_KF_with_cholesky(xf, Pf, dz, Q, Hf)

    particle.landPos[lm_id, :2] = xf.T
    particle.landCov[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle

def compute_jacobians(particle, xf, Pf, Q):
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    d2 = dx**2 + dy**2
    d = math.sqrt(d2)

    zp = np.array(
        [d, norm_angle(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)


    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q

    return zp, Hf, Sf

def update_KF_with_cholesky(xf, Pf, v, Q, Hf):
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q

    S = (S + S.T) * 0.5
    SChol = np.linalg.cholesky(S).T
    SCholInv = np.linalg.inv(SChol)
    W1 = PHt @ SCholInv
    W = W1 @ SCholInv.T

    x = xf + W @ v
    P = Pf - W1 @ W1.T

    return x, P

def normalize_weight(particles):

    sumw = sum([p.weight for p in particles])

    try:
        for i in range(NUM_PARTICLES):
            particles[i].weight /= sumw
    except ZeroDivisionError:
        for i in range(NUM_PARTICLES):
            particles[i].weight = 1.0 / NUM_PARTICLES

        return particles

    return particles

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def resampling(particles):#low variance re-sampling

    particles = normalize_weight(particles)

    pw = []
    for i in range(NUM_PARTICLES):
        pw.append(particles[i].weight)

    pw = np.array(pw)

    Neff = 1.0 / (pw @ pw.T)  # Effective particle number
    #print(Neff)


    inds = [] 
    if Neff < NUM_PARTICLES/1.5:  # resampling
        wcum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / NUM_PARTICLES) - 1 / NUM_PARTICLES

        resampleid = base + np.random.rand(base.shape[0]) / NUM_PARTICLES

        inds = []
        ind = 0
        for ip in range(NUM_PARTICLES):
            while ((ind < wcum.shape[0] - 1) and (resampleid[ip] > wcum[ind])):
                ind += 1
            inds.append(ind)

        tparticles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tparticles[inds[i]].x
            particles[i].y = tparticles[inds[i]].y
            particles[i].yaw = tparticles[inds[i]].yaw
            particles[i].weight = 1.0 / NUM_PARTICLES

    return particles


if __name__=="__main__":
    particles = [Particle() for i in range(NUM_PARTICLES)]
    u = np.zeros((2,1))
    global history 
    history = [] 
    global initial_time
    initial_time = None
    global odom_hist
    odom_hist = []

    global total
    total = 0
    global missed
    missed = 0

    try:
        rospy.init_node('OurfastSLAM')
        subscriber_odom = OdomSubscriber('/odom', particles=particles, u=u)
        subscriber_aruco = ArucoSubscriber('/fiducial_transforms', particles=particles, u=u)
        print("alive")


        # plt.ion()  # Turn on interactive mode
        # plt.show()  # Show the plot window


        rospy.spin()
    except:
        rospy.loginfo("Got interrupted request!")