import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Define camera parameters as global variables
CAMERA_MATRIX = np.array([[490.100238, 0.000000, 325.559625],
                          [0.000000, 489.477285, 240.754324],
                          [0.000000, 0.000000, 1.000000]], dtype=np.float32)

DIST_COEFFS = np.array([0.122115, -0.212824, 0.000813, 0.002590, 0.000000], dtype=np.float32)

# Define marker size as a global variable
MARKER_SIZE = 0.05  # Replace with your actual marker size

class ArucoDetector:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('aruco_detector', anonymous=True)

        # Create a CvBridge object
        self.bridge = CvBridge()

        # Initialize the ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()

        # Subscribe to the input compressed image topic
        self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_callback)

        # Publisher for the processed image
        self.image_pub = rospy.Publisher('/camera/aruco_detected', Image, queue_size=10)

    def image_callback(self, msg):
        try:
            # Convert the ROS CompressedImage message to an OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Extract the header from the compressed image message
        header = msg.header
        timestamp = header.stamp.to_sec()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        # If markers are detected
        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)

            for i in range(len(ids)):
                # Draw the detected markers and their axes
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)


                # Calculate distance and bearing
                tvec = tvecs[i][0]
                distance = np.linalg.norm(tvec)
                bearing = np.arctan2(tvec[0], tvec[2])

                # Print the distance, bearing, and timestamp
                print(f"Timestamp: {timestamp:.3f} s, Marker ID: {ids[i][0]}, Distance: {distance:.2f} meters, Bearing: {np.degrees(bearing):.2f} degrees")

        # Convert the OpenCV image back to ROS Image message and publish it
        try:
            processed_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            processed_image_msg.header = header  # Preserve the header from the original message
            self.image_pub.publish(processed_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        # Optionally display the frame
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        detector = ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()


# # Draw axis
# if hasattr(cv2.aruco, 'drawAxis'):
#     cv2.aruco.drawAxis(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], 0.1)
# else:
#     # Custom implementation of drawAxis if not available
#     self.custom_draw_axis(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[i], tvecs[i], 0.1)

# def custom_draw_axis(self, image, camera_matrix, dist_coeffs, rvec, tvec, length):
#     """ Draw custom axis on the image """
#     points = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]).reshape(-1, 3)
#     axis_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
#     axis_points = axis_points.reshape(-1, 2)
#     origin = tuple(axis_points[3].ravel().astype(int))
#     image = cv2.line(image, origin, tuple(axis_points[0].ravel().astype(int)), (0, 0, 255), 3)
#     image = cv2.line(image, origin, tuple(axis_points[1].ravel().astype(int)), (0, 255, 0), 3)
#     image = cv2.line(image, origin, tuple(axis_points[2].ravel().astype(int)), (255, 0, 0), 3)
