import numpy as np
import math
from copy import deepcopy

from ipywidgets import *
import matplotlib.pyplot as plt



#SIM
SIM_TIME = 80.0 
MAX_RANGE = 8.0
RSIM = np.diag([.3, np.deg2rad(1.0)])**2 
QSIM = np.diag([.3, np.deg2rad(1.0)])**2 

#fastSLAM
NUM_PARTICLES = 100
NUM_LANDMARKS = 10
DT = 0.1 #deta t
R = np.diag([1.5, np.deg2rad(10.0)])**2 
Q = np.diag([1.5, np.deg2rad(10.0)])**2 

class Particle:
    def __init__(self):
        self.x = 0
        self.y = -10
        self.yaw = 0

        self.weight = 1/NUM_PARTICLES

        self.landPos = np.zeros((NUM_LANDMARKS, 2)) # 2 pq é 2D
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

def observation(xTrue, u, posLandmarks):

    xTrue = motion_model(xTrue, u)

    z = -np.ones((3, len(posLandmarks[:, 0])))
    for i in range(len(posLandmarks[:, 0])):

        dx = posLandmarks[i, 0] - xTrue[0, 0]
        dy = posLandmarks[i, 1] - xTrue[1, 0]
        d = math.sqrt(dx**2 + dy**2)
        angle = norm_angle(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * QSIM[0, 0]  # add noise
            anglen = angle + np.random.randn() * QSIM[1, 1]  # add noise
            zi = np.array([dn, norm_angle(anglen), i]).reshape(3, 1) #distance, angle, id
            z[:, i] = zi.squeeze() 
    #print(z)

    return xTrue, z

def update_with_observation(particles, z): #for landamarks that give the id 
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

    return particles, inds




if __name__=="__main__":
    posLandmarks = np.array([[10.0, -2.0],
                [15.0, 10.0],
                [3.0, 15.0],
                [-5.0, 20.0],
                [-5.0, 5.0],
                [-10.0, 15.0],
                [-15.0, 0.0],
                [-10.0, -10.0],
                [-5.0, -15.0],
                [5.0, -10.0]]) #landmarks
    first = True
    num_inter = 1
    odometry_rmse = []
    fastslam_rmse = []
    for var in range(num_inter):
        particles = [Particle() for i in range(NUM_PARTICLES)]
        u = np.array([[2.0, 0.2]]).reshape(2,1) #delta x, delta yaw
        ud1 = u[0, 0] + np.random.randn() * RSIM[0, 0]
        ud2 = u[1, 0] + np.random.randn() * RSIM[1, 1]
        udt = np.array([ud1, ud2]).reshape(2, 1)
    


        time = 0.0 
        history = [] 


        xTrue = np.zeros((3, 1)) # 3 = x,y,yaw
        xDR = np.zeros((3, 1))
        

        pos = Particle()
        part = np.zeros((3,1)) #x,y,yaw
        part[0] = pos.x
        part[1] = pos.y
        part[2] = pos.yaw
        poshist = []
        gtruth = []
        gtruthu = u
        part2 = np.zeros((3,1)) #x,y,yaw
        part2[0] = pos.x
        part2[1] = pos.y
        part2[2] = pos.yaw
        history = []
        
        while SIM_TIME >= time:
            
            for i in range(NUM_PARTICLES):
                sample_pose(particles[i], udt)#udt
            xTrue, z = observation(xTrue, u, posLandmarks)
            particles = update_with_observation(particles, z)
            
            particles, new_indices = resampling(particles)
            history.append(deepcopy(particles))
            time += DT


            #!noisy u for ploting
            nois_u = udt + (np.random.randn(1,2) @ R).T
            part = motion_model(part,nois_u)
            poshist.append(deepcopy(part))

            #!ground truth for plotinghow 
            part2 = motion_model(part2, gtruthu)
            gtruth.append(deepcopy(part2))

        #! Calculate RMSE for each time step
        
        cleanhistory = []
        cleanhistory = np.zeros((2,1)) 
        for t in range(len(history)):

            odometry_error = np.sqrt(np.mean((np.array(gtruth[t][:2]) - np.array(poshist[t][:2]))**2))

            maxwj = 0
            for j in range(len(history[t])):
                if(particles[j].weight > particles[maxwj].weight):
                    maxwj = j

            cleanhistory[0] = history[t][maxwj].x
            cleanhistory[1] = history[t][maxwj].y
            fastslam_error = np.sqrt(np.mean((np.array(gtruth[t][:2]) - np.array(cleanhistory[:2]))**2)) 
            
            if(first):
                odometry_rmse.append(odometry_error)
                fastslam_rmse.append(fastslam_error)
            else:
                odometry_rmse[t] += odometry_error
                fastslam_rmse[t] += fastslam_error
        first = False
    
    #!dividir por numero de loops
    for t in range(len(history)):
        odometry_rmse[t] /= num_inter
        fastslam_rmse[t] /= num_inter

    # Plotting the RMSE over time
    plt.figure()
    plt.plot(range(len(history)), odometry_rmse, label='Odometry')
    plt.plot(range(len(history)), fastslam_rmse, label='FastSLAM')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('Odometry vs FastSLAM RMSE')
    plt.legend()
    plt.show()

    #! Ploting  

    #!plot best particle
    pltparticles = []
    for i in range(len(history)):
        maxwj = 0
        for j in range(len(history[i])):
            if(particles[j].weight > particles[maxwj].weight):
                maxwj = j
        pltparticles.append(history[i][maxwj])   
    for particle in pltparticles:
        plt.scatter(particle.x, particle.y, color='green', s=9)
    # x_values = [particle.x for particle in pltparticles]
    # y_values = [particle.y for particle in pltparticles]
    # plt.plot(x_values, y_values, color='green')

    #plt.scatter(pltparticles[-1].landPos[:, 0], pltparticles[-1].landPos[:, 1], c='orange', label='Landmarks', s=30)
    plt.scatter(posLandmarks[:, 0], posLandmarks[:, 1], c='orange', label='Landmarks', s=30)


    #!plot only odom with noise
    for i in range(len(poshist)):
        plt.scatter(poshist[i][0], poshist[i][1], color='blue', s=9)
    # x_values = [poshist[i][0] for i in range(len(poshist))]
    # y_values = [poshist[i][1] for i in range(len(poshist))]
    # plt.plot(x_values, y_values, color='blue')

    
    
    #!plot only ground truth
    for i in range(len(gtruth)):
        plt.scatter(gtruth[i][0], gtruth[i][1], color='red', s=9)
    # x_values = [gtruth[i][0] for i in range(len(gtruth))]
    # y_values = [gtruth[i][1] for i in range(len(gtruth))]
    # plt.plot(x_values, y_values, color='red')

    
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Particles History')
    plt.grid(True)
    plt.axis('equal')

    plt.show()