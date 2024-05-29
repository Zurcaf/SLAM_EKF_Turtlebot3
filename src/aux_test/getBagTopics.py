import rosbag

try:
    # Open the ROS bag file
    bag = rosbag.Bag('../rosbags/data/29-04-2024_17-43-03.bag')

    # Get the topics from the bag file
    topics = bag.get_type_and_topic_info().topics.keys()

    print(f"Rosbag Topics:")

    # Print the topics
    for topic in topics:
        print(topic)

    # Close the bag when finished
    bag.close()

except rosbag.ROSBagException as e:
    print(f"Error opening the ROS bag file: {e}")
except FileNotFoundError:
    print("File not found. Check the path or filename.")
