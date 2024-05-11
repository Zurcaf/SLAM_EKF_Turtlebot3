import rospy
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import tf
import numpy as np

def pose_callback(msg):
    """
    Callback function for the /pose topic.

    Parameters:
        msg (Path): Message containing the pose data.
    """
    # Extract x, y, and theta from the pose message
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    # Assuming the orientation is represented as quaternion, converting it to euler angles
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quaternion)
    theta = euler[2]

    # Plot the robot pose
    plot_robot_pose(x, y, theta)

def plot_robot_pose(x, y, theta):
    """
    Plot the pose of a robot with an arrow indicating its direction.

    Parameters:
        x (float): x-coordinate of the robot's pose.
        y (float): y-coordinate of the robot's pose.
        theta (float): orientation of the robot in radians (0 is pointing along the x-axis).
    """
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plot the robot position
    ax.plot(x, y, 'ro')

    # Calculate the end point of the arrow
    arrow_length = 0.5
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)

    # Plot the arrow
    ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Pose')

    # Show the plot
    plt.show()

def main():
    # Initialize the ROS node
    rospy.init_node('localization_node')

    # Define the topic and message type
    listen_topic_pose = "/pose"

    # Subscribe to the /pose topic
    rospy.Subscriber(listen_topic_pose, Odometry, pose_callback)

    # Spin to keep the node running
    rospy.spin()

if __name__ == "__main__":
    main()
