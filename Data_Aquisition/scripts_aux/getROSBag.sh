#!/bin/bash

# Check if the argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 [TURTLEBOT_IP_LAST_DIGIT]"
    exit 1
fi

# Form the full IP address
TURTLEBOT_IP="192.168.28.$1"

# Get IP address of your PC
PC_IP=$(hostname -I)

# Function to edit .bashrc file
edit_bashrc() {
    echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
    echo "export TURTLEBOT3_NAME=waffle5" >> ~/.bashrc
    echo "export TURTLEBOT3_IP=$TURTLEBOT_IP" >> ~/.bashrc
    echo "export TURTLEBOT3_NUMBER=$1" >> ~/.bashrc
    echo "export ROS_MASTER_URI=http://$TURTLEBOT_IP:11311" >> ~/.bashrc
    echo "export ROS_HOSTNAME=$PC_IP" >> ~/.bashrc
    echo "export ROS_IP=$PC_IP" >> ~/.bashrc
    echo "Updated .bashrc file with TurtleBot3 configurations."
}

# Edit .bashrc file
gedit ~/.bashrc &
edit_bashrc

# Wait for the user to save the file
read -p "Press Enter after saving .bashrc..."

# Function to SSH into the TurtleBot3 and execute commands
run_ssh_commands() {
    ssh user@$TURTLEBOT_IP "$1"
    sleep 5  # Wait for 5 seconds before proceeding
}

# Start roscore on the TurtleBot3
run_ssh_commands 'roscore'

# Sync time and launch robot drivers on the TurtleBot3
run_ssh_commands 'sudo ntpdate ntp.ubuntu.com && roslaunch turtlebot3_bringup turtlebot3_robot.launch'

# Wait for the user to start recording
read -p "Press Enter to start recording rosbag..."

# Start recording topics in rosbags on the TurtleBot3
run_ssh_commands 'rosbag record -a'

# Wait for the user to stop recording
read -p "Press Enter after recording rosbag..."

# Copy the rosbag from the TurtleBot3 to the specified directory on the PC
scp user@$TURTLEBOT_IP:/home/user/*.bag /home/afonso/Desktop/SLAM_EKF_Turtlebot3/ROS_Bags/

echo "Rosbag copied to /home/afonso/Desktop/SLAM_EKF_Turtlebot3/ROS_Bags/"

