import numpy as np

class EKF_SLAM:
    def __init__(self, num_landmarks, initial_pose, initial_covariance, process_noise, measurement_noise):
        self.num_landmarks = num_landmarks
        self.state_dim = 3 + 2 * num_landmarks  # [x, y, theta, l1_x, l1_y, l2_x, l2_y, ..., ln_x, ln_y]
        self.state = np.array(initial_pose)
        self.covariance = np.array(initial_covariance)
        self.Q = np.array(process_noise)
        self.R = np.array(measurement_noise)
        
    def predict(self, delta_t, motion_command):
        # Motion model: simple odometry model
        v, w = motion_command
        x, y, theta = self.state[:3]
        theta += w * delta_t
        x += v * np.cos(theta) * delta_t
        y += v * np.sin(theta) * delta_t
        self.state[:3] = [x, y, theta]

        # Jacobian of motion model
        F = np.eye(self.state_dim)
        F[0, 2] = -v * np.sin(theta) * delta_t
        F[1, 2] = v * np.cos(theta) * delta_t

        # Expand process noise covariance matrix to match state dimension
        Q_expanded = np.zeros((self.state_dim, self.state_dim))
        Q_expanded[:3, :3] = self.Q

        # Update covariance using motion model and process noise
        self.covariance = F @ self.covariance @ F.T + Q_expanded



        
    def update(self, measurements):
        for measurement in measurements:
            landmark_id, range_observation, bearing_observation = measurement
            
            # Calculate expected measurement
            landmark_index = 3 + 2 * landmark_id
            landmark_x, landmark_y = self.state[landmark_index:landmark_index+2]
            delta_x = landmark_x - self.state[0]
            delta_y = landmark_y - self.state[1]
            expected_range = np.sqrt(delta_x**2 + delta_y**2)
            expected_bearing = np.arctan2(delta_y, delta_x) - self.state[2]
            
            # Jacobian of observation model
            H = np.zeros((2, self.state_dim))
            H[:, :3] = [[-delta_x/expected_range, -delta_y/expected_range, 0],
                        [delta_y/(expected_range**2), -delta_x/(expected_range**2), -1]]
            H[:, landmark_index:landmark_index+2] = [[delta_x/expected_range, delta_y/expected_range],
                                                      [-delta_y/(expected_range**2), delta_x/(expected_range**2)]]
            
            # Kalman gain
            K = self.covariance @ H.T @ np.linalg.inv(H @ self.covariance @ H.T + self.R)
            
            # Update state and covariance
            innovation = np.array([range_observation - expected_range, bearing_observation - expected_bearing])
            self.state += K @ innovation
            self.covariance = (np.eye(self.state_dim) - K @ H) @ self.covariance
            
    def get_state(self):
        return self.state

# Example usage
num_landmarks = 2
initial_pose = [0, 0, 0]  # [x, y, theta]
initial_covariance = np.diag([1, 1, 0.1])  # Initial uncertainty
process_noise = np.diag([0.01, 0.01, 0.01])  # Process noise covariance
measurement_noise = np.diag([0.1, 0.1])  # Measurement noise covariance

ekf_slam = EKF_SLAM(num_landmarks, initial_pose, initial_covariance, process_noise, measurement_noise)

# Example motion command
delta_t = 0.1
motion_command = [0.1, 0.05]

# Example measurements [landmark_id, range, bearing]
measurements = [[0, 1.1, 0.1], [1, 1.5, -0.2]]

# Prediction step
ekf_slam.predict(delta_t, motion_command)

# Resize covariance matrix after prediction to account for additional landmarks
num_new_landmarks = ekf_slam.num_landmarks
state_dim_with_landmarks = 3 + 2 * num_new_landmarks
if state_dim_with_landmarks > ekf_slam.state_dim:
    ekf_slam.covariance = np.pad(ekf_slam.covariance, ((0, state_dim_with_landmarks - ekf_slam.state_dim), (0, state_dim_with_landmarks - ekf_slam.state_dim)), mode='constant', constant_values=0)

# Update step
ekf_slam.update(measurements)

# Get updated state
updated_state = ekf_slam.get_state()
print("Updated state:", updated_state)
