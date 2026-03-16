import sys
import tf
import tf2_ros
import numpy as np
import moveit_commander
import cv2
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_geometry_msgs
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from visualization_msgs.msg import Marker, MarkerArray
from base_optimization.msg import multi_target_pose
from moveit_msgs.msg import MoveGroupAction, Grasp
from interbotix_xs_msgs.msg import JointSingleCommand
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from base_optimization.srv import images_giver, images_giverResponse
import pyrealsense2 as rs
from tf.transformations import euler_from_quaternion
import threading
from std_msgs.msg import Bool

from base_optimization.dynamic_obstacle_monitor import DynamicObstacleMonitor

# How often to poll move_base state and check for obstacles
OBSTACLE_CHECK_RATE_HZ = 10

# Camera frustum parameters (Intel RealSense D435)
CAMERA_HFOV_DEG = 87.0
FRUSTUM_RANGE_M = 2.0


# Re-optimization synchronization state
waiting_for_reopt = False
reopt_poses = []
reopt_lock = threading.Lock()
reopt_done_event = threading.Event()
reopt_success = False


def add_new_pose(data):
    global waiting_for_reopt
    rospy.loginfo("received a new optimal base pose...")
    
    if waiting_for_reopt:
        # During re-optimization, collect poses in a separate list
        with reopt_lock:
            reopt_poses.append(data)
    else:
        opt_base_poses.append(data)
        opt_base_poses_history.append(data)
        viz_all_base_poses()
        viz_reachable_gripper_poses()


def reopt_status_callback(msg):
    """Called when find_opt_pose_multi publishes the optimization status."""
    global reopt_success
    if waiting_for_reopt:
        reopt_success = msg.data
        reopt_done_event.set()
    
def viz_all_base_poses():
    pose_array_msg = PoseArray()
    pose_array_msg.header.frame_id = "octomap_frame"
    pose_array_msg.header.stamp = rospy.Time.now()
    
    for multi_target in opt_base_poses_history:
        pose_array_msg.poses.append(multi_target.base_pose.pose)
    
    rospy.sleep(0.01)
    all_base_poses_pub.publish(pose_array_msg)

def viz_reachable_gripper_poses():
    """Publish all reachable gripper poses accumulated from optimization results."""
    pose_array_msg = PoseArray()
    pose_array_msg.header.frame_id = "octomap_frame"
    pose_array_msg.header.stamp = rospy.Time.now()
    
    for multi_target in opt_base_poses_history:
        for gp in multi_target.gripper_poses:
            pose_array_msg.poses.append(gp.pose)
    
    reachable_gripper_pub.publish(pose_array_msg)

def request_reoptimization(cancelled_target):
    """Request re-optimization for the gripper poses associated with a
    cancelled base pose target.  Publishes the gripper poses on /gripper_poses
    and waits for the optimization result from find_opt_pose_multi.
    If successful, the newly computed base poses are inserted at the front
    of the queue so they are executed immediately.
    If the optimization fails, nothing is inserted and the normal flow
    continues."""
    global waiting_for_reopt, reopt_success

    # Prepare a PoseArray with the gripper poses from the cancelled target
    reopt_pose_array = PoseArray()
    reopt_pose_array.header.stamp = rospy.Time.now()
    reopt_pose_array.header.frame_id = "octomap_frame"
    for ee_pose in cancelled_target.gripper_poses:
        reopt_pose_array.poses.append(ee_pose.pose)

    rospy.loginfo("Sending %d gripper poses for re-optimization...",
                  len(reopt_pose_array.poses))

    # Enter re-optimization mode
    with reopt_lock:
        waiting_for_reopt = True
        del reopt_poses[:]
        reopt_done_event.clear()
        reopt_success = False

    # Publish the gripper poses to trigger a new optimization
    reopt_gripper_pub.publish(reopt_pose_array)

    # Wait for the optimization to finish (timeout = 180s to be safe)
    rospy.loginfo("Waiting for re-optimization to complete...")
    reopt_done_event.wait()

    # Exit re-optimization mode
    with reopt_lock:
        waiting_for_reopt = False
        collected_poses = list(reopt_poses)
        del reopt_poses[:]

    if reopt_success and len(collected_poses) > 0:
        rospy.loginfo(
            "\u2713 Re-optimization succeeded. Inserting %d new base poses "
            "at the front of the queue.", len(collected_poses))
        # Insert in reverse order at position 0 so they maintain their
        # original sequence and are executed first
        for pose in reversed(collected_poses):
            opt_base_poses.insert(0, pose)
            opt_base_poses_history.append(pose)
        viz_all_base_poses()
        viz_reachable_gripper_poses()
    else:
        rospy.logwarn(
            "\u2717 Re-optimization failed or timed out. "
            "Continuing with the normal queue.")

# initialize the MoveIt stack
moveit_commander.roscpp_initialize(sys.argv)

# create a new ros node
rospy.init_node("move_to_next_pose")
rospy.loginfo("Created node 'move_to_next_pose'")

# subscribe to a topic to receive the newly found pose for the mobile base
rospy.Subscriber("add_opt_base_pose", multi_target_pose, add_new_pose)
rospy.loginfo("Created topic 'add_opt_base_pose'")

# subscribe to the optimization status topic
rospy.Subscriber("/optimization_status", Bool, reopt_status_callback)

# publisher to request re-optimization by sending gripper poses
reopt_gripper_pub = rospy.Publisher("/gripper_poses", PoseArray, queue_size=10)

# Instantiate the dynamic obstacle monitor (subscribes to projected map
# and segmented mask internally)
obstacle_monitor = DynamicObstacleMonitor()
rospy.loginfo("Dynamic obstacle monitor ready")

# TF buffer for getting robot pose
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

# Frustum parameters
hfov_rad = np.deg2rad(CAMERA_HFOV_DEG)

# publish on topic to move the base
move_base_client = actionlib.SimpleActionClient(
    "/locobot/move_base", MoveBaseAction)
move_base_client.wait_for_server()

nxt_EE_pose_pub = rospy.Publisher("/des_EE_pose", PoseStamped, queue_size=10)

# publish on a topic to visualize all the base poses
all_base_poses_pub = rospy.Publisher("/all_base_poses", PoseArray, queue_size=10)

# publish the current base target being navigated to
current_base_target_pub = rospy.Publisher("/locobot/current_base_target", PoseStamped, queue_size=1, latch=True)

# publish on topics to visualize gripper target poses in RViz
gripper_targets_pub = rospy.Publisher("/gripper_target_poses", PoseArray, queue_size=10)
current_gripper_target_pub = rospy.Publisher("/current_gripper_target", Marker, queue_size=10)

# publish ALL reachable gripper poses (accumulated across all received base poses)
reachable_gripper_pub = rospy.Publisher("/reachable_gripper_poses", PoseArray, queue_size=10, latch=True)

# create a list of poses (queue)
opt_base_poses = []
opt_base_poses_history = []

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

arm_group = moveit_commander.MoveGroupCommander("interbotix_arm", wait_for_servers=30.0)
arm_group.set_max_velocity_scaling_factor(0.8)
arm_group.set_max_acceleration_scaling_factor(0.8)
arm_group.set_end_effector_link("locobot/ee_gripper_link")
# arm_group.set_planner_id("RRTstar")

# plan to Insert_backpack pose
# arm_group.set_named_target("Insert_backpack")
# success = False
# tries = 0
# while success == False and tries < 3:
#     rospy.loginfo("Trying to reach the backpack")
#     rospy.sleep(1)
#     success = arm_group.go(wait=True)
#     tries = tries + 1
# arm_group.stop()


rospy.loginfo("Waiting for a new optimal base pose...")

# start looping on all the desired poses
while not rospy.is_shutdown():
    # check for empy
    if len(opt_base_poses) <= 0:
        rospy.sleep(2)
    else:
        # extract the first element (oldest inserted)
        nxt_base_pose = opt_base_poses.pop(0)

        # publish the current target for visualization
        current_target_msg = PoseStamped()
        current_target_msg.header.stamp = rospy.Time.now()
        current_target_msg.header.frame_id = 'octomap_frame'
        current_target_msg.pose = nxt_base_pose.base_pose.pose
        current_base_target_pub.publish(current_target_msg)

        # create a MoveBaseGoal message
        goal_base_pose = MoveBaseGoal()

        goal_base_pose.target_pose.header.stamp = rospy.Time.now()
        goal_base_pose.target_pose.header.frame_id = 'octomap_frame'

        goal_base_pose.target_pose.pose = nxt_base_pose.base_pose.pose

        rospy.sleep(5)
        
        # Send goal and poll for result while checking for dynamic obstacles
        goal_reached = False
        goal_cancelled = False
        nav_start_time = rospy.Time.now()
        while not rospy.is_shutdown() and not goal_reached and not goal_cancelled:
            rospy.loginfo("Moving the base to the optimal pose...")
            goal_base_pose.target_pose.header.stamp = rospy.Time.now()
            move_base_client.send_goal(goal_base_pose)

            check_rate = rospy.Rate(OBSTACLE_CHECK_RATE_HZ)
            while not rospy.is_shutdown():
                state = move_base_client.get_state()

                # Goal finished successfully
                if state == actionlib.GoalStatus.SUCCEEDED:
                    nav_elapsed = (rospy.Time.now() - nav_start_time).to_sec()
                    rospy.logwarn("Base navigation time: %.3f seconds", nav_elapsed)
                    rospy.loginfo("Goal finished and succeeded.")
                    goal_reached = True
                    break

                # Goal was aborted or rejected by move_base itself
                if state in (actionlib.GoalStatus.ABORTED,
                             actionlib.GoalStatus.REJECTED):
                    rospy.logwarn("Goal aborted/rejected by move_base, retrying...")
                    break  # will re-send the same goal

                # Goal still in progress → check for dynamic obstacles
                if state in (actionlib.GoalStatus.ACTIVE,
                             actionlib.GoalStatus.PENDING):
                    # Get current robot pose
                    try:
                        tf_stamped = tf_buffer.lookup_transform(
                            "octomap_frame", "locobot/base_footprint", rospy.Time(0),
                            timeout=rospy.Duration(0.1))
                        t = tf_stamped.transform.translation
                        q = tf_stamped.transform.rotation
                        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                        robot_pose = (t.x, t.y, yaw)
                        
                        # Check if target is obstructed (only within frustum)
                        if obstacle_monitor.is_target_obstructed(
                                nxt_base_pose.base_pose.pose,
                                robot_pose=robot_pose,
                                hfov_rad=hfov_rad,
                                frustum_range=FRUSTUM_RANGE_M):
                            
                            move_base_client.cancel_goal()
                            rospy.logwarn(
                                "Target appears obstructed. Waiting 5 s to confirm...")
                            rospy.sleep(5.0)

                            # Re-acquire robot pose for the second check
                            try:
                                tf_stamped2 = tf_buffer.lookup_transform(
                                    "octomap_frame", "locobot/base_footprint", rospy.Time(0),
                                    timeout=rospy.Duration(0.1))
                                t2 = tf_stamped2.transform.translation
                                q2 = tf_stamped2.transform.rotation
                                _, _, yaw2 = euler_from_quaternion([q2.x, q2.y, q2.z, q2.w])
                                robot_pose2 = (t2.x, t2.y, yaw2)
                            except (tf2_ros.LookupException,
                                    tf2_ros.ConnectivityException,
                                    tf2_ros.ExtrapolationException):
                                robot_pose2 = robot_pose  # fallback

                            if obstacle_monitor.is_target_obstructed(
                                    nxt_base_pose.base_pose.pose,
                                    robot_pose=robot_pose2,
                                    hfov_rad=hfov_rad,
                                    frustum_range=FRUSTUM_RANGE_M):
                                # Still obstructed → determine obstacle type
                                if obstacle_monitor.is_obstacle_mobile(nxt_base_pose.base_pose.pose):
                                    rospy.logwarn(
                                        "Dynamic obstacle confirmed at target (within frustum)! "
                                        "It is a mobile agent. Cancelling and re-queuing target at end.")
                                    # Mobile agent: put the same target back at the end of the queue
                                    opt_base_poses.append(nxt_base_pose)
                                    # opt_base_poses_history.append(nxt_base_pose)
                                    viz_all_base_poses()
                                else:
                                    rospy.logwarn(
                                        "Dynamic obstacle confirmed at target (within frustum)! "
                                        "It is a static obstacle. Requesting re-optimization...")
                                    # Static obstacle: request re-optimization for the
                                    # gripper poses that this base pose was supposed to reach
                                    request_reoptimization(nxt_base_pose)
                                
                                goal_cancelled = True
                                break
                            else:
                                # Obstacle cleared → re-send the same goal
                                rospy.loginfo(
                                    "Obstacle cleared after waiting. Re-sending goal...")
                                goal_base_pose.target_pose.header.stamp = rospy.Time.now()
                                move_base_client.send_goal(goal_base_pose)
                                # Continue polling loop
                    except (tf2_ros.LookupException,
                            tf2_ros.ConnectivityException,
                            tf2_ros.ExtrapolationException) as e:
                        rospy.logwarn_throttle(5.0, "TF lookup failed: %s", str(e))

                check_rate.sleep()

        if goal_cancelled:
            continue  # skip arm motion, go to next pose in queue

        # move the arm towards the desired targets
        # Publish all gripper target poses for this base pose
        gripper_pose_array = PoseArray()
        gripper_pose_array.header.frame_id = "octomap_frame"
        gripper_pose_array.header.stamp = rospy.Time.now()
        for gp in nxt_base_pose.gripper_poses:
            gripper_pose_array.poses.append(gp.pose)
        gripper_targets_pub.publish(gripper_pose_array)

        for idx, ee_pose in enumerate(nxt_base_pose.gripper_poses):
            rospy.sleep(2)
            # pick_and_store(ee_pose.pose, nxt_base_pose.base_yaw)
            ee_pose.header.stamp = rospy.Time.now()

            # Publish a marker for the current gripper target being reached
            marker = Marker()
            marker.header.frame_id = "octomap_frame"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "current_gripper_target"
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = ee_pose.pose
            marker.scale.x = 0.12
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0)
            current_gripper_target_pub.publish(marker)

            arm_group.set_pose_target(ee_pose)
            arm_group.set_planning_time(10.0)
            success = False
            tries = 0
            ee_start_time = rospy.Time.now()
            while success == False and tries < 5:
                rospy.loginfo("Trying to reach gripper target (attempt %d/5)", tries + 1)
                success = arm_group.go(wait=True)
                tries = tries + 1
                if not success and tries < 5:
                    rospy.sleep(1)
            arm_group.stop()
            arm_group.clear_pose_targets()
            ee_elapsed = (rospy.Time.now() - ee_start_time).to_sec()
            if success:
                rospy.logwarn("EE target %d reached in %.3f seconds", idx, ee_elapsed)
            else:
                rospy.logwarn("EE target %d FAILED after %.3f seconds (5 attempts)", idx, ee_elapsed)

        rospy.sleep(2)
        arm_group.set_named_target("Insert_backpack")
        arm_group.set_planning_time(10.0)
        success = False
        tries = 0
        while success == False and tries < 5:
            rospy.loginfo("Trying to reach Insert_backpack (attempt %d/5)", tries + 1)
            success = arm_group.go(wait=True)
            tries = tries + 1
            if not success and tries < 5:
                rospy.sleep(1)
        arm_group.stop()
        arm_group.clear_pose_targets()
        if not success:
            rospy.logwarn("Failed to reach Insert_backpack after 5 attempts")
