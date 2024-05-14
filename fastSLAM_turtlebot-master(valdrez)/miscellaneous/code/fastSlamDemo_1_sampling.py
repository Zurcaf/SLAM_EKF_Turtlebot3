import numpy as np
import math
from copy import deepcopy

from ipywidgets import *
import matplotlib.pyplot as plt

NUM_PARTICLES = 100
NUM_LANDMARKS = 10
DT = 0.1 #deta t
SIM_TIME = 50.0 #tempo de simulação

R = np.diag([2.0, np.deg2rad(2.0)])**2 #uma covariancia escolhida por nos -> 3m e 10º

class Particle:
    def __init__(self):#?, numLandmarks):
        self.x = 0
        self.y = 0
        self.yaw = 0

        self.weight = 1/NUM_PARTICLES

        self.landPos = np.zeros((NUM_LANDMARKS, 2)) # 2 pq é 2D
        #?self.landCov = np.zeros((numLandmarks*2, 2)) # 2 pq é 2D

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


if __name__=="__main__":
    '''
    Y = []
    (...)
    for i in range(NUM_PARTICLES):#verificar se da loop por tudo o que é preciso
        X = sample_pose(particles[i], u)
    '''

    particles = [Particle() for i in range(NUM_PARTICLES)]
    u = np.array([[1.0, 0.1]]).reshape(2,1) #velocidade*velocidade angular
    time = 0.0
    history = []

    while SIM_TIME >= time:
        for i in range(NUM_PARTICLES):
            sample_pose(particles[i], u)
        history.append(deepcopy(particles))
        time += DT
    
    #! Ploting  
    for particle in history[0]:
        plt.scatter(particle.x, particle.y, color='green')
    for particle in history[100]:
        plt.scatter(particle.x, particle.y, color='orange')
    for particle in history[200]:
        plt.scatter(particle.x, particle.y, color='red')


    linear_speed = u[0] / DT
    angular_speed = u[1] / DT

    theta = np.linspace(0, 2*np.pi, 100)
    x = linear_speed * np.cos(theta) 
    y = linear_speed * np.sin(theta)+10

    plt.plot(x, y, color='blue', label='Linear and Angular Speed Circle')
    
    
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Particles History')
    plt.grid(True)
    plt.show()


