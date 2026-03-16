# import sys
# import tf
# import tf2_ros
# import numpy as np
# import moveit_commander
# import cv2
# import geometry_msgs.msg
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# import tf2_geometry_msgs
# import rospy
# import actionlib
# from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped
# from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from visualization_msgs.msg import Marker
# from moveit_msgs.msg import MoveGroupAction, Grasp
# from interbotix_xs_msgs.msg import JointSingleCommand
# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge
# import pyrealsense2 as rs

# # initialize the MoveIt stack
# moveit_commander.roscpp_initialize(sys.argv)

# # create a new ros node
# rospy.init_node("move_to_next_pose")
# rospy.loginfo("Created node 'move_to_next_pose'")


# # subscribe to the topic to command single joints
# joint_sing_comm = rospy.Publisher(
#     "/locobot/commands/joint_single", JointSingleCommand, queue_size=10)


# gripper_pub = rospy.Publisher(
#     "/locobot/gripper_controller/command", JointTrajectory, queue_size=10)

# robot = moveit_commander.RobotCommander()
# scene = moveit_commander.PlanningSceneInterface()

# arm_group = moveit_commander.MoveGroupCommander("interbotix_arm")
# arm_group.set_max_velocity_scaling_factor(0.8)
# arm_group.set_max_acceleration_scaling_factor(0.8)

# gripper_group = moveit_commander.MoveGroupCommander("interbotix_gripper")
# gripper_group.set_max_velocity_scaling_factor(1)
# gripper_group.set_max_acceleration_scaling_factor(0.8)

# # add a collision object (box)
# coords = [0.0503, 0.1305, 0.6814]

# object_pose = PoseStamped()
# object_pose.header.frame_id = "map"
# object_pose.pose.position.x = coords[0]
# object_pose.pose.position.y = coords[1]
# object_pose.pose.position.z = coords[2]
# object_pose.pose.orientation.x = 0.0
# object_pose.pose.orientation.y = 0.0
# object_pose.pose.orientation.z = 0.0
# object_pose.pose.orientation.w = 1.0

# scene.add_box(name="object", pose=object_pose, size=(0.03, 0.03, 0.03))
# rospy.sleep(1)

# # compute the grap pose orientation
# # roll = 0
# # pitch = 0
# # yaw = yaw_angle
# # # rospy.logwarn(yaw)

# # quat = tf.transformations.quaternion_from_euler(
# #         roll, pitch, yaw, axes='sxyz')
# quat = [0.0582, 0.4997, 0.8264, 0.2531]

# # create a Grasp pipeline message
# grasp_msg = Grasp()

# # set the grasp pose
# grasp_msg.grasp_pose.header.frame_id = "map"
# grasp_msg.grasp_pose.pose.position.x = coords[0]
# grasp_msg.grasp_pose.pose.position.y = coords[1]
# grasp_msg.grasp_pose.pose.position.z = coords[2]
# grasp_msg.grasp_pose.pose.orientation.x = quat[0]
# grasp_msg.grasp_pose.pose.orientation.y = quat[1]
# grasp_msg.grasp_pose.pose.orientation.z = quat[2]
# grasp_msg.grasp_pose.pose.orientation.w = quat[3]

# # set the pre-grasp approach
# grasp_msg.pre_grasp_approach.direction.header.frame_id = "/locobot/base_footprint"
# grasp_msg.pre_grasp_approach.direction.vector.x = 1.0
# grasp_msg.pre_grasp_approach.direction.vector.y = 0.0
# grasp_msg.pre_grasp_approach.direction.vector.z = 0.0
# grasp_msg.pre_grasp_approach.min_distance = 0.05
# grasp_msg.pre_grasp_approach.desired_distance = 0.10

# # set the post-grasp approach
# grasp_msg.post_grasp_retreat.direction.header.frame_id = "/locobot/base_footprint"
# grasp_msg.post_grasp_retreat.direction.vector.x = 0.0
# grasp_msg.post_grasp_retreat.direction.vector.y = 0.0
# grasp_msg.post_grasp_retreat.direction.vector.z = 1.0
# grasp_msg.post_grasp_retreat.min_distance = 0.05
# grasp_msg.post_grasp_retreat.desired_distance = 0.10

# # allow the contact with the object
# grasp_msg.allowed_touch_objects = ["object"]

# # set the gripper fingers as open before grasping
# # open_gripper(grasp_msg)
# # open_gripper(None)

# # set gripper as closed during grasping
# # close_gripper(grasp_msg)

# # plan to a gripper pose
# # arm_group.set_position_target([ee_pose.position.x, ee_pose.position.y, ee_pose.position.z])
# # ee_pose.orientation.x = quat[0]
# # ee_pose.orientation.y = quat[1]
# # ee_pose.orientation.z = quat[2]
# # ee_pose.orientation.w = quat[3]
# # for visualization and debug purposes


# # open_gripper()

# # try to pick the object
# rospy.loginfo("Trying to reach x:{}, y:{}, z:{}".format(
#     coords[0], coords[1], coords[2]))
# pick_success = arm_group.pick(object_name="object", grasp=grasp_msg)
        
import rospy
from geometry_msgs.msg import PoseStamped


rospy.init_node("grasp_pose_visualizer")
rospy.loginfo("Created node")

pred_grasp_pub = rospy.Publisher("/predicted_grasping_pose", PoseStamped, queue_size=10)
rospy.loginfo("Publisher created")

pose_stmp_msg = PoseStamped()
pose_stmp_msg.header.frame_id = "map"
pose_stmp_msg.header.stamp = rospy.Time.now()

pose_stmp_msg.pose.position.x = 0.0201
pose_stmp_msg.pose.position.y = 0.0883
pose_stmp_msg.pose.position.z = 0.6471
pose_stmp_msg.pose.orientation.x = 0.7120
pose_stmp_msg.pose.orientation.y = 0.6761
pose_stmp_msg.pose.orientation.z = 0.1894
pose_stmp_msg.pose.orientation.w = -0.0052

rospy.sleep(0.5)
pred_grasp_pub.publish(pose_stmp_msg)

rospy.spin()
 


 