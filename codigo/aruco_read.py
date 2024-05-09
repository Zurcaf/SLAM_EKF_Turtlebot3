import cv2
import cv2.aruco as aruco

# Load the camera calibration parameters if available
camera_matrix = None
dist_coefficients = None

# Load camera calibration parameters if available
try:
    with open('camera_calibration.txt', 'r') as f:
        data = f.readlines()
        camera_matrix = np.array(eval(data[0]))
        dist_coefficients = np.array(eval(data[1]))
except FileNotFoundError:
    print("Camera calibration file not found. Make sure to calibrate your camera first.")

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define the ArUco dictionary
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

        # Initialize the detector parameters using the default values
        parameters = aruco.DetectorParameters_create()

        # Detect the markers in the image
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # Draw the detected markers and their IDs on the frame
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Print the IDs of the detected markers
            print("Detected ArUco IDs:", ids)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
