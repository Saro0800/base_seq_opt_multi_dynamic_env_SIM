import rospy
import tf
import tf.transformations
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# create a rosnode
rospy.init_node("des_EE_pose_publisher", anonymous=True)

# define a the desired points
des_points = []
des_points.append([4.15+0.3, 0.45, 0.2])
des_points.append([4.15+0.2, 0.45, 0.2])
des_points.append([4.15+0.1, 0.45, 0.2])
des_points.append([4.15, 0.4-0.1, 0.2])    # corner
des_points.append([4.15, 0.45-0.2, 0.2])
# des_points.append([4, -0.75, 0.3])
# des_points.append([4, -0.1, 0.25])
# des_points.append([4, -0.75+0.5, 0.15])
# des_points.append([4, -1.+0.5, 0.4])
# des_points.append([4, -1.25, 0.2])


# create a Marker message
des_EE_pose_multi = Marker()
des_EE_pose_multi.header.frame_id = "map"
des_EE_pose_multi.header.stamp = rospy.Time.now()

des_EE_pose_multi.type = 7
des_EE_pose_multi.action = des_EE_pose_multi.ADD
des_EE_pose_multi.id = 0

des_EE_pose_multi.scale.x = 0.02
des_EE_pose_multi.scale.y = 0.02
des_EE_pose_multi.scale.z = 0.02

des_EE_pose_multi.pose.orientation.x = 0.0
des_EE_pose_multi.pose.orientation.y = 0.0
des_EE_pose_multi.pose.orientation.z = 0.0
des_EE_pose_multi.pose.orientation.w = 1.0

for p in des_points:
    des_EE_pose_multi.points.append(Point(p[0], p[1], p[2]))
    des_EE_pose_multi.colors.append(ColorRGBA(0.0, 0.0, 1.0, 1.0))

# create a publisher to publish the desired end-effector pose
des_EE_topic_multi = rospy.Publisher("/des_EE_pose_multi", Marker, queue_size=10)

rospy.loginfo("Publishing the desired end-effector pose...")
# publish the desired end-effector pose
rospy.sleep(1)
des_EE_topic_multi.publish(des_EE_pose_multi)
# rospy.spin()
# while not rospy.is_shutdown():
#     rate.sleep()