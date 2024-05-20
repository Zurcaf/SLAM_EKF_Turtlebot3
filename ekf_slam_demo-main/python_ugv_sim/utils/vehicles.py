'''
Script for vehicle classes and functions
'''
import pygame
from pygame.locals import Rect
import numpy as np
import scipy

class Robot:
    '''
    Parent class for robots
    '''
    length = 0.3
    width = 0.3
    def __init__(self):
        pass
    def get_corners(self):
        '''
        Return robot polygon corners
        '''
        x = self.get_pose()
        # Get the corners that define the robot
        corners = np.zeros((2,4))
        # corners in robot frame
        corners[0,:] = (self.length/2.0)*np.array([[1,1,-1,-1]])
        corners[1,:] = (self.width/2.0)*np.array([[1,-1,-1,1]])
        # Rotate corners
        theta = x[2]
        R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        corners = np.matmul(R,corners)
        # Offset corners
        corners = corners + np.array([[self.x[0]],[self.x[1]]])
        # Turn into list of points
        corners = [corners[:,i] for i in range(corners.shape[1])]
        return corners
    def get_pose(self):
        return None
    def get_position(self):
        return None

class DifferentialDrive(Robot):
    '''
    Robot with differential drive dynamics
    x = robot state
        x[0] = position x (m)
        x[1] = position y (m)
        x[2] = heading theta (rad)
    u = controls
        u[0] = v, forward velocity (m/s)
        u[1] = omega, angular velocity (rad/s)
    EOM = equations of motion
        xdot[0] = v*cos(theta)
        xdot[1] = v*sin(theta)
        xdot[2] = omega
    '''
    max_v = 2.0
    max_omega = 2.0
    def __init__(self,x_init):
        Robot.__init__(self)
        self.set_state(x_init)
    def move_step(self,u,dt):
        y = np.zeros(5)
        y[:3] = self.x; y[3:] = u
        result = scipy.integrate.solve_ivp(self.EOM,[0,dt],y)
        self.x = result.y[:3,-1]
        self.x[2] = np.arctan2(np.sin(self.x[2]),np.cos(self.x[2]))
    def EOM(self,t,y):
        px = y[0]; py = y[1]; theta = y[2]
        v = max(min(y[3],self.max_v),-self.max_v); omega = max(min(y[4],self.max_omega),-self.max_omega) # forward and angular velocity
        ydot = np.zeros(5)
        ydot[0] = v*np.cos(theta)
        ydot[1] = v*np.sin(theta)
        ydot[2] = omega
        ydot[3] = 0
        ydot[4] = 0
        return ydot
    def set_state(self,x):
        self.x = x
        self.x[2] = np.arctan2(np.sin(x[2]),np.cos(x[2])) # Keep angle between -pi and +pi
    def get_pose(self):
        return self.x
    def get_position(self):
        return self.x[0:2]
    def update_u(self,u,key_event):
        '''
        Update controls
        '''
        if key_event.type==pygame.KEYDOWN:
            if key_event.key==pygame.K_LEFT:
                u[1] = self.max_omega
            elif key_event.key==pygame.K_RIGHT:
                u[1] = -1*self.max_omega
            if key_event.key==pygame.K_UP:
                u[0] = self.max_v
            elif key_event.key==pygame.K_DOWN:
                u[0] = -1*self.max_v
        if key_event.type==pygame.KEYUP:
            if key_event.key==pygame.K_LEFT:
                u[1] = 0
            elif key_event.key==pygame.K_RIGHT:
                u[1] = 0
            if key_event.key==pygame.K_UP:
                u[0] = 0
            elif key_event.key==pygame.K_DOWN:
                u[0] = 0
        return u