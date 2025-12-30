import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# create a rosnode
rospy.init_node("des_EE_pose_publisher", anonymous=True)

# define a the desired points
des_points = []
des_points.append([-0.908767, 0.143703, 0.230574])
des_points.append([-0.890545, 0.213786, 0.230502])
des_points.append([-0.919003, 0.280877, 0.230501])
des_points.append([-1.08169, 0.517985, 0.230508])
des_points.append([-1.13586, 0.555097, 0.23051])
des_points.append([-1.21794, 0.540532, 0.230495])
des_points.append([-1.01722, 2.45538, 0.230505])
des_points.append([-1.01275, 2.51664, 0.230505])
des_points.append([-0.786005, 4.96004, 0.320559])
des_points.append([-0.843832, 4.95477, 0.320522])
des_points.append([-1.03006, 5.09174, 0.370565])
des_points.append([-1.01939, 5.14843, 0.370568])
des_points.append([0.132438, 3.256509, 0.220499])
des_points.append([0.136797, 3.323965, 0.220499])
# des_points.append([])
# des_points.append([])


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