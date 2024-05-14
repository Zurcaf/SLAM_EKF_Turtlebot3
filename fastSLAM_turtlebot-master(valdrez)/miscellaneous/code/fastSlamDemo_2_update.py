import numpy as np
import math
from copy import deepcopy

from ipywidgets import *
import matplotlib.pyplot as plt

NUM_PARTICLES = 50
NUM_LANDMARKS = 10


DT = 0.1 #deta t
SIM_TIME = 10.0 #tempo de simulação

MAX_RANGE = 200.0 #max range do sensor #!only for simulation
OFFSET_YAWRATE_NOISE = 0.01 #!only for simulation???

R = np.diag([3.0, np.deg2rad(10.0)])**2 #uma covariancia escolhida por nos -> 3m e 10º
Q = np.diag([0.5, np.deg2rad(10.0)])**2 #uma covariancia escolhida por nos -> 0.5m e 10º

class Particle:
    def __init__(self):#?, numLandmarks):
        self.x = 0
        self.y = 0
        self.yaw = 0

        self.weight = 1/NUM_PARTICLES

        self.landPos = np.zeros((NUM_LANDMARKS, 2)) # 2 pq é 2D
        #self.landPos[:, 2] = -1
        self.landCov = np.zeros((NUM_LANDMARKS*2, 2)) # 2 pq é 2D

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

def motion_model(x, u):#basicameente pega na particula e aplica odom
    F = np.array([[1,0,0],
                [0,1,0],
                [0,0,1]])

    B = np.array([[DT*math.cos(x[2,0]), 0],
                [DT*math.sin(x[2,0]), 0],
                [0,DT]])
    
    x = F@x + B@u


    x[2, 0] = norm_angle(x[2, 0])

    return x

def norm_angle(angle): 
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def observation(xTrue, xd, u, posLandmarks):
    #! change for real data!!!
    # gets the postition of the robot with odom
    xTrue = motion_model(xTrue, u)

    z = -np.ones((3, len(posLandmarks[:, 0])))
    for i in range(len(posLandmarks[:, 0])):#!the camera does this dor us

        dx = posLandmarks[i, 0] - xTrue[0, 0]
        dy = posLandmarks[i, 1] - xTrue[1, 0]
        d = math.sqrt(dx**2 + dy**2)
        angle = norm_angle(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q[0, 0]  # add noise
            anglen = angle + np.random.randn() * Q[1, 1]  # add noise
            zi = np.array([dn, norm_angle(anglen), i]).reshape(3, 1) #distance, angle, id
            z[:, i] = zi.squeeze() #basicamente lista de observações
            
        
    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * R[0, 0]
    ud2 = u[1, 0] + np.random.randn() * R[1, 1] + OFFSET_YAWRATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud

def update_with_observation(particles, z): #!for landamarks that give the id 
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

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

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

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

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

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q

    return zp, Hv, Hf, Sf

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



#!helper
def plot_particles(particles, posLandmarks):
    # Extract particle positions and weights
    particle_positions = np.array([[particle.x, particle.y] for particle in particles])
    particle_weights = np.array([particle.weight for particle in particles])

    # Plot landmarks
    plt.scatter(posLandmarks[:, 0], posLandmarks[:, 1], c='red', label='Landmarks')
    plt.scatter(particle_positions[:,0], particle_positions[:,1], c="green")

    # Plot particle landPos with weights
    for particle, weight in zip(particles, particle_weights):
        particle_landmarks = particle.landPos[particle.landPos[:, 0] != 0]  # Exclude zero-initialized landmarks
        if len(particle_landmarks) > 0:
            num_landmarks = len(particle_landmarks)
            colors = np.full(num_landmarks, weight)  # Create 1D array of color values for each point
            plt.scatter(particle_landmarks[:, 0], particle_landmarks[:, 1], c=colors, cmap='viridis', vmin=0, vmax=0.01)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Particle Filter')
    plt.colorbar(label='Weights')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_particles2(particles, posLandmarks):
    # Extract particle positions and weights
    particle_positions = np.array([[particle.x, particle.y] for particle in particles])
    particle_weights = np.array([particle.weight for particle in particles])
    particle_indices = np.arange(len(particles))

    plt.scatter(particle_positions[:, 0], particle_positions[:, 1], c=particle_weights, cmap='viridis')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Particle Filter')
    plt.colorbar(label='Weights')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_particles3(particles, part):
    # Extract particle positions and weights
    particle_positions = np.array([[particle.x, particle.y] for particle in particles])
    particle_weights = np.array([particle.weight for particle in particles])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # Plot landmarks on the first subplot
    ax1.scatter(particle_positions[:, 0], particle_positions[:, 1], c='blue')

    ax1.scatter(part[0], part[1], c='red')

    # Add indices as text labels for each particle on the first subplot
    for i, (x, y) in enumerate(particle_positions):
        ax1.text(x, y, str(i), color='black', ha='center', va='center')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Particle Filter')
    ax1.grid(True)

    # Plot the particle weight labels on the second subplot
    ax2.axis('off')
    weight_labels = ['Particle {} - Weight: {}'.format(i, weight) for i, weight in enumerate(particle_weights)]
    weight_label = '\n'.join(weight_labels)
    ax2.text(0, 0.5, weight_label, verticalalignment='center', fontsize=12)

    plt.tight_layout()
    plt.show()






if __name__=="__main__":
    posLandmarks = np.array([[10.0, -2.0],
                [15.0, 10.0],
                [3.0, 15.0],
                [-5.0, 20.0],
                [-5.0, 5.0],
                [-20.0, 15.0],
                [-15.0, -5.0],
                [-10.0, -10.0],
                [-5.0, -15.0],
                [5.0, -20.0]]) #landmarks

    particles = [Particle() for i in range(NUM_PARTICLES)]
    u = np.array([[1.0, 0.1]]).reshape(2,1) #velocidade*velocidade angular 

    time = 0.0 #?
    history = [] #?

    xTrue = np.zeros((3, 1)) # 3 = x,y,yaw
    xDR = np.zeros((3, 1))
    xTrue, z, _, ud = observation(xTrue, xDR, u, posLandmarks)

    # Initialize landmarks
    particles = update_with_observation(particles, z) 
    print("Initial weight (only one particle): ", particles[0].weight)
    print("position: " + str(particles[0].x) + " " + str(particles[0].y) + " " + str(particles[0].yaw))


    pos = Particle()
    part = np.zeros((3,1)) #x,y,yaw
    part[0] = pos.x
    part[1] = pos.y
    part[2] = pos.yaw

    while SIM_TIME >= time:
        for i in range(NUM_PARTICLES):
            sample_pose(particles[i], u)
        xTrue, z, _, ud = observation(xTrue, xDR, u, posLandmarks)
        history.append(deepcopy(particles))
        time += DT

        # part = motion_model(part,u)
    
    # pos.x = part[0,0]
    # pos.y = part[1,0]
    # pos.yaw = part[2,0]

    particles = update_with_observation(particles, z)

    plot_particles3(particles, xTrue)

    # #! Ploting  
    # for particle in history[0]:
    #     plt.scatter(particle.x, particle.y, color='green')
    # for particle in history[50]:
    #     plt.scatter(particle.x, particle.y, color='orange')
    # for particle in history[100]:
    #     plt.scatter(particle.x, particle.y, color='red')


    # linear_speed = u[0] / DT
    # angular_speed = u[1] / DT

    # theta = np.linspace(0, 2*np.pi, 100)
    # x = linear_speed * np.cos(theta) 
    # y = linear_speed * np.sin(theta)+10

    # plt.plot(x, y, color='blue', label='Linear and Angular Speed Circle')
    
    
    # plt.xlabel('X position')
    # plt.ylabel('Y position')
    # plt.title('Particles History')
    # plt.grid(True)
    # plt.show()

    #TODO:HERE
    #ver se estamos sempre a ignoar quando é preciso (0,0) e id = -1