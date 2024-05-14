import rospy
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
from tf.transformations import euler_from_quaternion



class OdomSubscriber:
    def __init__(self, topic_name):
        self.sub = rospy.Subscriber(topic_name, Odometry, self.Callback)
    
    def Callback(self, msg):
        #rospy.loginfo("FastCallback %f", msg.pose.pose.position.y)
        x_positions.append(msg.pose.pose.position.x)
        y_positions.append(msg.pose.pose.position.y)
        quaternions.append(msg.pose.pose.orientation)
        print("x: " + str(msg.pose.pose.position.x) + " y: " + str(msg.pose.pose.position.y) + " yaw: "+ str(euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))[2]))

        yaw_values = [euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))[2] for quat in quaternions]



        plt.clf()
        plt.quiver(x_positions, y_positions, np.cos(yaw_values), np.sin(yaw_values), angles='xy', scale_units='xy', scale=100)
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Robot Movement')
        plt.grid(True)
        plt.pause(0.001)

    
if __name__=="__main__":
    x_positions = []
    y_positions = []
    quaternions = []
    try:
        rospy.init_node('bag_subscriber')
        subscriber_odom = OdomSubscriber('/odom')

        plt.ion()  # Turn on interactive mode
        plt.show()  # Show the plot window

        rospy.spin() #basicamente um loop que continua a correr o main ate fecharmos o programa

    except rospy.ROSInterruptException:
        rospy.loginfo("Got interrupted request!")
    

    