import sys
import rospy
import numpy as np
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def open_gripper():
    global gripper_pub
    
    point = JointTrajectoryPoint()
    point.positions = [0.5]
    point.velocities = [1]
    point.accelerations = [0.5]
    point.time_from_start.secs = 1
    
    
    joint_trajectory = JointTrajectory()
    joint_trajectory.joint_names = ["left_finger"]
    joint_trajectory.points.append(point)
    
    rospy.sleep(0.1)
    gripper_pub.publish(joint_trajectory)
    rospy.sleep(2)
    
def close_gripper():
    global gripper_pub
    
    point = JointTrajectoryPoint()
    point.positions = [-0.2]
    point.velocities = [1]
    point.accelerations = [0.5]
    point.time_from_start.secs = 1
    
    
    joint_trajectory = JointTrajectory()
    joint_trajectory.joint_names = ["left_finger"]
    joint_trajectory.points.append(point)
    
    rospy.sleep(0.1)
    gripper_pub.publish(joint_trajectory)
    rospy.sleep(2)

    

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("mov_braccio_completo", anonymous=True)
print("qui")
gripper_pub = rospy.Publisher("/locobot/gripper_controller/command", JointTrajectory, queue_size=10)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

arm_group = moveit_commander.MoveGroupCommander("interbotix_arm")
arm_group.set_max_velocity_scaling_factor(0.8)
arm_group.set_max_acceleration_scaling_factor(0.8)

gripper_group = moveit_commander.MoveGroupCommander("interbotix_gripper")
gripper_group.set_max_velocity_scaling_factor(1)
gripper_group.set_max_acceleration_scaling_factor(0.8)

open_gripper()
# gripper_group.set_named_target("Home")
# gripper_group.go(wait=True)
# gripper_group.stop()

assert(False)

# plan to a gripper pose
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.w = 1.0
pose_goal.position.x = 0.3
pose_goal.position.y = 0.1
pose_goal.position.z = 0.3

arm_group.set_pose_target(pose_goal)
success = False
while success==False:
    rospy.sleep(1)
    success = arm_group.go(wait=True)
arm_group.stop()
arm_group.clear_pose_targets()

rospy.sleep(2)

close_gripper()
# gripper_group.set_named_target("Closed")
# gripper_group.go(wait=True)
# gripper_group.stop()

# plan to Insert_backpack pose
arm_group.set_named_target("Insert_backpack")
success = False
while success==False:
    rospy.sleep(1)
    success = arm_group.go(wait=True)
arm_group.stop()

open_gripper()
# gripper_group.set_named_target("Home")
# gripper_group.go(wait=True)
# gripper_group.stop()

# # plan to the Sleep pose
# arm_group.set_named_target("Sleep")
# success = False
# while success==False:
#     rospy.sleep(1)
#     success = arm_group.go(wait=True)
# arm_group.stop()



