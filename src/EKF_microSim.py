import numpy as np
import matplotlib.pyplot as plt

# Generate some example 2D landmarks (you can replace this with your own data)
landmarks = np.array([[2, 3], [5, 7], [8, 4], [10, 9]])

# Initial robot pose (x, y, heading angle in radians)
initial_pose = np.array([3, 2.5, np.pi/4])

# EKF-SLAM parameters (you can adjust these)
R = np.diag([0.1, 0.1])        # Measurement noise covariance
Q = np.diag([0.1, 0.1, 0.01])  # Process noise covariance

# Initialize state vector (robot pose + landmark positions)
state = np.hstack((initial_pose, landmarks.flatten()))

# Initialize covariance matrix
P = np.eye(3 + 2 * len(landmarks))

# Simulate sensor measurements (range and bearing to landmarks)
# In practice, you'd replace this with actual sensor data
measurements = np.array([[2.5, np.pi/6], [4.0, np.pi/4], [3.8, -np.pi/3]])

# EKF-SLAM main loop
for z in measurements:
    # Prediction step (motion model)
    # Assuming a simple motion model for demonstration
    # For example, if the robot moves forward with a constant velocity, the prediction step would be:
    # state[:3] += [v * np.cos(state[2]), v * np.sin(state[2]), omega] * dt
    # where v is the linear velocity, omega is the angular velocity, and dt is the time step

    # Measurement update step
    H = np.zeros((2, 3 + 2 * len(landmarks)))
    for i, landmark in enumerate(landmarks):
        dx = landmark[0] - state[0]
        dy = landmark[1] - state[1]
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)
        H[:, :3] = [[-sqrt_q * dx, -sqrt_q * dy, 0], [dy, -dx, -q]]
        H[:, 3 + 2*i:3 + 2*i + 2] = [[sqrt_q * dx, sqrt_q * dy], [-dy, dx]]

    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    state += K @ (z - np.array([np.sqrt(q), np.arctan2(dy, dx) - state[2]]))
    P = (np.eye(len(state)) - K @ H) @ P

# Extract updated robot pose and landmark positions
updated_pose = state[:3]
updated_landmarks = state[3:].reshape(-1, 2)

# Visualization (similar to your code)
plt.figure(figsize=(8, 6))
plt.scatter(updated_landmarks[:, 0], updated_landmarks[:, 1], marker="*", color="r", label="Landmarks")
plt.plot(updated_pose[0], updated_pose[1], "bo", label="Updated Robot Pose")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D EKF-SLAM Example")
plt.grid(True)
plt.show()

