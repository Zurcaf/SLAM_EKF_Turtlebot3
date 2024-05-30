import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import time
import rospy


# Function to read camera parameters from a file
def read_camera_params(filename):                                           # mudar isto tudo para acomodar ROS  #
    with open(filename) as f:                                               #
        loadeddict = yaml.safe_load(f)                                      #----------------------------------- #
        cam_matrix = np.array(loadeddict.get('camera_matrix'))              #
        dist_coeffs = np.array(loadeddict.get('dist_coeff'))                #----------------------------------- #
    return cam_matrix, dist_coeffs                                          #
                                                                            #----------------------------------- #
# Load camera parameters                                                    #
cam_matrix, dist_coeffs = read_camera_params('tutorial_camera_params.yml')  # mudar isto tudo para acomodar ROS  #

# Set coordinate system for the marker
marker_length = 0.05  # Example marker length in meters
obj_points = np.array([
    [-marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

# Initialize the ArUco dictionary and detector parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

#ROSSSSSSSSSSSSSSSSSSSSSSSSSS

# Capture video
video = ''  # Provide the path to the video file if any             # mudar isto tudo para acomodar ROS  #
cam_id = 0  # Camera ID, usually 0 for the default camera           # mudar isto tudo para acomodar ROS  #
input_video = cv2.VideoCapture(video if video else cam_id)          # mudar isto tudo para acomodar ROS  #
wait_time = 0 if video else 10                                      # mudar isto tudo para acomodar ROS  #

#ROSSSSSSSSSSSSSSSSSSSSSSSSSSS
total_time = 0
total_iterations = 0

estimate_pose = True
show_rejected = False

pub = = rospy-Publisher('ARUCO_topic', int, queue size-10)
rospy.init_node('publisher_node', anonymous=True)
rate = rospy.Rate(0.1)
rospy.loginfo("Publisher Node Started")

while input_video.isOpened():
    ret, image = input_video.read()
    if not ret:
        break

    start_time = time.time()

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    
    if estimate_pose and ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, cam_matrix, dist_coeffs)

    current_time = time.time() - start_time
    total_time += current_time
    total_iterations += 1

    if total_iterations % 30 == 0:
        print(f"Detection Time = {current_time * 1000:.2f} ms (Mean = {1000 * total_time / total_iterations:.2f} ms)")

    # Draw results
    image_copy = image.copy()
    if ids is not None:
        aruco.drawDetectedMarkers(image_copy, corners, ids)
        if estimate_pose:
            for rvec, tvec in zip(rvecs, tvecs):
                aruco.drawAxis(image_copy, cam_matrix, dist_coeffs, rvec, tvec, marker_length * 1.5)

    if show_rejected and rejected is not None:
        aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(100, 0, 255))

    cv2.imshow("out", image_copy)
    key = cv2.waitKey(wait_time)
    if key == 27:  # Exit if ESC is pressed
        break
    
    range = np.linalg.norm(tvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    # ENVIAR ROS THINGS PARA OS SUBSCRITOS RANGE BEARING E ID  (ids,range,yaw)
    pub.publish(range)
    pub.publish(yaw)
    pub.publish(ids)

input_video.release()
cv2.destroyAllWindows()
