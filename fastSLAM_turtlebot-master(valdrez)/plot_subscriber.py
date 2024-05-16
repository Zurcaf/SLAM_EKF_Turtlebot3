#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
#from fiducial_msgs.msg import FiducialTransformArray
#from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import matplotlib.pyplot as plt
import numpy as np

class PlotSubscriber:
    def __init__(self, topic_name):
        self.sub = rospy.Subscriber(topic_name, Float64MultiArray, self.Callback)
        self.array = []

        # def Callback(self, msg):
        #     self.array.append(msg.data)
        #     plt.clf()
        #     for i in range(len(self.array)):
        #         plt.scatter(self.array[i][0], self.array[i][1], color='green')
        #         odd = True
        #         for j in range(100000):
        #             if j <=1:
        #                 continue
        #             if odd:
        #                 try:
        #                     odd = False
        #                     plt.scatter(self.array[i][j], self.array[i][j+1], color='orange')
        #                 except:
        #                     break
        #             else:
        #                 odd = True
        #     plt.xlabel('X position')
        #     plt.ylabel('Y position')
        #     plt.title('Particles History')
        #     plt.grid(True)
        #     plt.pause(0.0001)
    
    def Callback(self, msg):
        global x_position
        global y_position

        global odomposx
        global odomposy
        global slamposx
        global slamposy
        #print(msg)

        # global tfposx
        # global tfposy
        # global tflandmarkx
        # global tflandmarky

        # global tfcontx
        # global tfconty
        

        global time #!remove
        time += 1 #!remove
        
        global initialx
        global initialy

        if(time%10==0):
            plt.clf()
            print(time)
            initialx = x_position
            initialy = y_position
            
            odomposx.append(x_position)
            odomposy.append(y_position)
            slamposx.append(msg.data[0])
            slamposy.append(msg.data[1])
            plt.scatter(slamposx, slamposy, color='green', s=5)
            plt.scatter(odomposx, odomposy, color='blue', s=5)

            j=2
            while True:
                try:
                    plt.scatter(msg.data[j], msg.data[j+1], color='orange', s=10)
                    plt.text(msg.data[j], msg.data[j+1], str(int(msg.data[j+4])), color='black', ha='center', va='center')
                    theta = np.linspace(0, 2*np.pi, 100)
                    x = msg.data[j] + msg.data[j+2] * np.cos(theta)
                    y = msg.data[j+1] + msg.data[j+3] * np.sin(theta)
                    plt.plot(x, y, color='red')
                    j+=5
                except:
                    break
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Particles History')
        plt.grid(True)
        plt.axis('equal')


        # plt.xlim(42, 50)#!c, z1
        # plt.ylim(30, 43)
        # plt.xlim(45, 50)#!c, f1
        # plt.ylim(19, 30)
        # plt.xlim(45, 51)#!c, f0
        # plt.ylim(12, 20)
        # plt.xlim(-30, 30)#!
        # plt.ylim(-30, 30)#!
        # plt.xlim(45, 55)#!l
        # plt.ylim(10, 20)


        plt.pause(0.0001)
        

class OdomSubscriber:
    def __init__(self, topic_name):
        self.sub = rospy.Subscriber(topic_name, Odometry, self.Callback)
    
    def Callback(self, msg):
        global x_position
        global y_position
        x_position = msg.pose.pose.position.x
        y_position = msg.pose.pose.position.y

class FiducialSubscriber:
    def __init__(self, topic_name):
        self.sub = rospy.Subscriber(topic_name, PoseWithCovarianceStamped, self.Callback)
    
    def Callback(self, msg):
        # print(msg)
        # print("-----------------------------")
        # print(msg.pose.pose.position.x)
        # print(msg.pose.pose.position.y)
        # print(msg.pose.pose.position.z)
        # print("-----------------------------")
        global tfposx
        global tfposy
        global initialx
        global initialy
        # global tflandmarkx
        # global tflandmarky
        # if(msg.transforms[0].child_frame_id == "base_footprint"):
        tfposx = msg.pose.pose.position.z #+ initialx
        tfposy = msg.pose.pose.position.x #+ initialy
        # else:
        #     tflandmarkx.append(msg.transforms[0].transform.translation.x)
        #     tflandmarky.append(msg.transforms[0].transform.translation.y)



if __name__=="__main__":
    global x_position
    x_position = 0
    global y_position
    y_position = 0

    global time
    time = 0

    global odomposx
    odomposx = []
    global odomposy
    odomposy = []
    global slamposx
    slamposx = []
    global slamposy
    slamposy = []

    # global tfposx
    # tfposx = 0
    # global tfposy
    # tfposy = 0
    # global tflandmarkx
    # tflandmarkx = []
    # global tflandmarky
    # tflandmarky = []

    # global tfcontx
    # tfcontx = []
    # global tfconty
    # tfconty = []

    global initialx
    initialx = 0
    global initialy
    initialy = 0


    rospy.init_node('plot_subscriber')
    subscriber_plot = PlotSubscriber('/toplot')
    subscriber_odom = OdomSubscriber('/odom')
    subscriber_fiducial = FiducialSubscriber('/fiducial_pose')

    plt.ion()  # Turn on interactive mode
    plt.show()  # Show the plot window

    rospy.spin() 