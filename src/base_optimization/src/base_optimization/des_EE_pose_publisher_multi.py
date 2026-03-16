import rospy
import tf
import tf.transformations
import numpy as np
import json
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import ColorRGBA

# create a rosnode
rospy.init_node("des_EE_pose_publisher", anonymous=True)

# create a publisher topic
pub_pose = rospy.Publisher("tags_poses", PoseArray, queue_size=10)

# Load tags from JSON file
tags_dict = {}
json_file = 'tag_poses_GOOD.json'

try:
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Convert to desired format: integer keys and tuple values
    for tag_id, transform in json_data.items():
        tag_num = int(tag_id)  # Convert string key to integer
        position = transform['position']
        orientation = transform['orientation']
        tags_dict[tag_num] = (position, orientation)
    
    rospy.loginfo(f"Loaded {len(tags_dict)} tags from {json_file}")
    print(f"Tag IDs loaded: {sorted(tags_dict.keys())}")

except FileNotFoundError:
    rospy.logerr(f"File {json_file} not found!")
    rospy.signal_shutdown("JSON file not found")
    exit(1)
except Exception as e:
    rospy.logerr(f"Error loading JSON file: {e}")
    rospy.signal_shutdown("Error loading JSON")
    exit(1)

n_tags = len(tags_dict.keys())

pose_array_msg = PoseArray()
pose_array_msg.header.frame_id = "map"
pose_array_msg.header.stamp = rospy.Time.now()

for tag_id in sorted(tags_dict.keys()):
    print(tag_id)
    pose_msg = Pose()
    position, quat = tags_dict[tag_id]
    
    pose_msg.position.x = position[0]
    pose_msg.position.y = position[1]
    pose_msg.position.z = position[2]
    
    pose_msg.orientation.x = quat[0]
    pose_msg.orientation.y = quat[1]
    pose_msg.orientation.z = quat[2]
    pose_msg.orientation.w = quat[3]
    
    pose_array_msg.poses.append(pose_msg)


while not rospy.is_shutdown():
    rospy.sleep(0.5)
    pub_pose.publish(pose_array_msg)