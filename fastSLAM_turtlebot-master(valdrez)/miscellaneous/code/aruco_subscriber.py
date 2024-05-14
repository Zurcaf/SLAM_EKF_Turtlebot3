import rospy
from  fiducial_msgs.msg import FiducialTransformArray
import numpy as np

#/fiducial_transforms [fiducial_msgs/FiducialTransformArray]
 #/fiducial_vertices [fiducial_msgs/FiducialArray]


class ArucoSubscriberTransforms:
    def __init__(self, topic_name):
        self.sub = rospy.Subscriber(topic_name, FiducialTransformArray, self.fastCallback)
    
    def fastCallback(self, ros_data):
        print("New Frame")
        for i in range(10):
            try:
                x = float(ros_data.transforms[i].transform.translation.x)
                z = float(ros_data.transforms[i].transform.translation.z)
                id = int(ros_data.transforms[i].fiducial_id)
                #y = float(ros_data.transforms[i].transform.translation.y)
            except:
                break
            dist = np.sqrt(x**2 + z**2)
            tg_value = x/z
            angle = np.arctan(tg_value)
            angle = np.degrees(angle)
            # print("ID:" + str(id)+"\n")
            # print("Distance:"+ str(dist) + "\n")
            # print("Angle:"+ str(angle) + "\n")
            print(ros_data)

        







if __name__ == '__main__':
    rospy.init_node("aruco_subscriber")
    
    subscriber_fast = ArucoSubscriberTransforms("/fiducial_transforms")

    rospy.spin()

