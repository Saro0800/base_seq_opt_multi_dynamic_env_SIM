import sys
import tf
import tf2_ros
import numpy as np
import moveit_commander
import cv2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_geometry_msgs
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, PoseArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from base_optimization.msg import multi_target_pose
from moveit_msgs.msg import MoveGroupAction, Grasp
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from base_optimization.srv import images_giver, images_giverResponse
import pyrealsense2 as rs
from std_msgs.msg import Float64

ROI_SIZE = 100  # pixels

def set_pan_tilt_to_zero():
    # set tilt angle
    tilt_msg = Float64()
    tilt_msg.data = 0.0

    tilt_msg_pub.publish(tilt_msg)

    # set pan angle
    pan_msg = Float64()
    pan_msg.data = 0.0
    rospy.sleep(0.05)

    pan_msg_pub.publish(pan_msg)

def reset_pan_tilt():
    # set tilt angle
    tilt_msg = Float64()
    tilt_msg.data = 0.2618
    rospy.sleep(0.05)

    tilt_msg_pub.publish(tilt_msg)

    # set pan angle
    pan_msg = Float64()
    pan_msg.data = 0.0
    rospy.sleep(0.05)

    pan_msg_pub.publish(pan_msg)

def set_pan_tilt(ee_pose_original):
    frame = "map"

    tmp_pose = PoseStamped()
    tmp_pose.header.frame_id = frame
    tmp_pose.pose.position = ee_pose_original.position
    tmp_pose.pose.orientation.w = 1.0

    # request the transform
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_buffer.can_transform("locobot/camera_link", frame, rospy.Time(0))
    transform = tf_buffer.lookup_transform(
        "locobot/camera_link", frame, rospy.Time(0), rospy.Duration(1))
    transformed_msg = tf2_geometry_msgs.do_transform_pose(tmp_pose, transform)

    # find RPY angles
    coord_camera = np.array([transformed_msg.pose.position.x,
                             transformed_msg.pose.position.y,
                             transformed_msg.pose.position.z])
        
    # find yaw and pitch angle
    yaw = np.arctan2(coord_camera[1], coord_camera[0])
    pitch = np.arctan2(coord_camera[2], coord_camera[0])
    
    # set tilt angle
    tilt_msg = Float64()
    tilt_msg.data = -pitch
    rospy.sleep(0.05)

    tilt_msg_pub.publish(tilt_msg)

    # set pan angle
    pan_msg = Float64()
    pan_msg.data = yaw
    rospy.sleep(0.05)

    pan_msg_pub.publish(pan_msg)
    
    return pitch, yaw

def deproject_2d_to_3d(u, v, depth, fx, fy, cx, cy):
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    return np.array([x, y, z])

def camera_frame_to_map(refined_grasp):
    tmp_pose = PoseStamped()
    tmp_pose.header.frame_id = "locobot/camera_depth_link"
    tmp_pose.pose.position.x = refined_grasp[0]
    tmp_pose.pose.position.y = refined_grasp[1]
    tmp_pose.pose.position.z = refined_grasp[2]
    tmp_pose.pose.orientation.w = 1.0

    # request the transform
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_buffer.can_transform("locobot/camera_link", "locobot/camera_depth_link", rospy.Time(0))
    transform = tf_buffer.lookup_transform(
        "locobot/camera_link", "locobot/camera_depth_link", rospy.Time(0), rospy.Duration(1))
    transformed_msg = tf2_geometry_msgs.do_transform_pose(tmp_pose, transform)
    
    transformed_msg.pose.orientation.x = 0.0
    transformed_msg.pose.orientation.y = 0.0
    transformed_msg.pose.orientation.z = 0.0
    transformed_msg.pose.orientation.w = 1.0

    # request the transform to map
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_buffer.can_transform("map", "locobot/camera_link", rospy.Time(0))
    transform = tf_buffer.lookup_transform(
        "map", "locobot/camera_link", rospy.Time(0), rospy.Duration(1))
    transformed_msg = tf2_geometry_msgs.do_transform_pose(transformed_msg, transform)
    
    return transformed_msg

def correct_ee_pose(ee_pose_original):
    rospy.sleep(0.05)
    set_pan_tilt_to_zero()

    # set pan and tilt in a way to have the center
    # of the image aligned to original grasping pose
    rospy.sleep(0.05)
    pan, tilt = set_pan_tilt(ee_pose_original)

    rospy.sleep(0.5)
    bridge = CvBridge()
    rospy.wait_for_service("/locobot/images_giver_service")
    get_igms = rospy.ServiceProxy("/locobot/images_giver_service", images_giver)

    # collect the grasps from 10 frames
    detected_graps = []
    for i in range(2):
        rospy.sleep(0.05)
        srv_resp = get_igms()
        color_msg = srv_resp.color_image
        depth_msg = srv_resp.depth_image
        camera_info = srv_resp.camera_info

        # Convert images
        color_image = bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # set the center as original grasping point
        h, w, _ = color_image.shape
        u = w // 2
        v = h // 2
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        # ROI around the center
        x1 = max(u - ROI_SIZE // 2, 0)
        x2 = min(u + ROI_SIZE // 2, w)
        y1 = max(v - ROI_SIZE // 2, 0)
        y2 = min(v + ROI_SIZE // 2, h)
        roi = color_image[y1:y2, x1:x2]

        # Estimate background color as median color of ROI
        bg_color = np.median(roi.reshape(-1, 3), axis=0)

        # Calculate distance from background for each pixel in ROI
        color_dist = np.linalg.norm(roi.astype(np.float32) - bg_color, axis=2)

        # Normalize and threshold: keep only the most different regions, e.g., top 5%
        thresh_val = np.percentile(color_dist, 95)
        mask = (color_dist >= thresh_val).astype(np.uint8) * 255

        # Find all contours in the mask (regions most different from background)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis_img = color_image.copy()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis_img, (u, v), 7, (0, 0, 255), 2)
        cv2.putText(vis_img, "Original (center)", (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        if contours:
            # Find the largest region most different from background
            largest_region = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_region)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cx_full = cx + x1
                cy_full = cy + y1
                depth_val_corr = depth_image[cy_full, cx_full]
                if depth_val_corr == 0:
                    # Try a small neighborhood if depth is missing
                    neighborhood = depth_image[max(cy_full-2,0):min(cy_full+3,h),
                                               max(cx_full-2,0):min(cx_full+3,w)]
                    valid = neighborhood[neighborhood > 0]
                    if valid.size == 0:
                        print("No valid depth at centroid.")
                        continue
                    depth_val_corr = int(np.median(valid))
                depth_m_corr = depth_val_corr * 0.001

                intrinsics = rs.intrinsics()
                intrinsics.width = camera_info.width
                intrinsics.height = camera_info.height
                intrinsics.ppx = camera_info.K[2]
                intrinsics.ppy = camera_info.K[5]
                intrinsics.fx = camera_info.K[0]
                intrinsics.fy = camera_info.K[4]
                intrinsics.model = rs.distortion.none
                intrinsics.coeffs = [i for i in camera_info.D]

                refined_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx_full, cy_full], depth_m_corr)
                detected_graps.append(refined_3d)

                # Draw corrected grasping point
                cv2.circle(vis_img, (cx_full, cy_full), 7, (0, 255, 0), 2)
                cv2.putText(vis_img, "Corrected (most diff color)", (cx_full+5, cy_full-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                ros_image = bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
                rospy.sleep(0.05)
                img_pub.publish(ros_image)
            else:
                print("Largest region has zero area.")
        else:
            print("No region detected in ROI.")

        # create a pose stamped and publish it if needed here

    if not detected_graps:
        print("No refined grasp points detected. Returning None.")
        return None

    detected_graps = np.array(detected_graps)
    refined_grasp = np.mean(detected_graps, axis=0)
    rospy.loginfo(refined_grasp)

    # transform from the image reference frame to map
    refined_grasp_map = camera_frame_to_map(refined_grasp)
    rospy.loginfo(refined_grasp_map)

    # refined_grasp_map.pose.position.z = refined_grasp_map.pose.position.z + 0.01
    for i in range(5):
        rospy.sleep(0.05)
        pose_pub.publish(refined_grasp_map)
    
    reset_pan_tilt()
    
    return pan, tilt, refined_grasp_map.pose
    
def open_gripper(grasp_msg=None):
    global gripper_pub
    
    if grasp_msg is None:
        gripper_group.set_named_target("Open")
        success = False
        tries = 0
        while success == False and tries < 3:
            rospy.loginfo("Opening the gripper")
            rospy.sleep(0.05)
            success = gripper_group.go(wait=True)
            tries = tries + 1
        gripper_group.stop()
    else:
        grasp_msg.grasp_posture.joint_names = ["left_finger", "right_finger"]

        point = JointTrajectoryPoint()
        point.positions = [0.037, -0.037]
        point.velocities = [1, 1]
        point.accelerations = [0.5, 0.5]
        point.effort = [5, 5]
        point.time_from_start = rospy.Duration(1.0)

        grasp_msg.grasp_posture.points.append(point)
    
def close_gripper(grasp_msg=None):
    global gripper_pub

    if grasp_msg is None:
        gripper_group.set_named_target("Closed")
        success = False
        tries = 0
        while success == False and tries < 3:
            rospy.loginfo("Closing the gripper")
            rospy.sleep(0.05)
            success = gripper_group.go(wait=True)
            tries = tries + 1
        gripper_group.stop()
    else:
        grasp_msg.grasp_posture.joint_names = ["left_finger", "right_finger"]

        point = JointTrajectoryPoint()
        point.positions = [0.01, -0.01]
        point.velocities = [1, 1]
        point.accelerations = [1, 1]
        point.effort = [5, 5]
        point.time_from_start = rospy.Duration(1.0)

        grasp_msg.grasp_posture.points.append(point)    
    
def pick_and_store(ee_pose, yaw_angle):

    # refine the ee_grasp basing on the camera input
    # pan, tilt, new_ee_pose = correct_ee_pose(ee_pose)
    set_pan_tilt(ee_pose)

    # add a collision object (box)
    object_pose = PoseStamped()
    object_pose.header.frame_id = "map"
    object_pose.pose.position = ee_pose.position
    object_pose.pose.orientation.x = 0.0
    object_pose.pose.orientation.y = 0.0
    object_pose.pose.orientation.z = 0.0
    object_pose.pose.orientation.w = 1.0

    scene.add_box(name="object", pose=object_pose, size=(0.03, 0.03, 0.03))
    rospy.sleep(0.05)

    # compute the grap pose orientation
    roll = 0
    pitch = 0
    yaw = yaw_angle

    quat = tf.transformations.quaternion_from_euler(
        roll, pitch, yaw, axes='sxyz')

    # create a Grasp pipeline message
    grasp_msg = Grasp()

    # set the grasp pose
    grasp_msg.grasp_pose.header.frame_id = "map"
    grasp_msg.grasp_pose.pose.position = ee_pose.position
    grasp_msg.grasp_pose.pose.orientation.x = quat[0]
    grasp_msg.grasp_pose.pose.orientation.y = quat[1]
    grasp_msg.grasp_pose.pose.orientation.z = quat[2]
    grasp_msg.grasp_pose.pose.orientation.w = quat[3]

    # set the pre-grasp approach
    grasp_msg.pre_grasp_approach.direction.header.frame_id = "/locobot/base_footprint"
    grasp_msg.pre_grasp_approach.direction.vector.x = 1.0
    grasp_msg.pre_grasp_approach.direction.vector.y = 0.0
    grasp_msg.pre_grasp_approach.direction.vector.z = 0.0
    grasp_msg.pre_grasp_approach.min_distance = 0.05
    grasp_msg.pre_grasp_approach.desired_distance = 0.10

    # set the post-grasp approach
    grasp_msg.post_grasp_retreat.direction.header.frame_id = "/locobot/base_footprint"
    grasp_msg.post_grasp_retreat.direction.vector.x = 0.0
    grasp_msg.post_grasp_retreat.direction.vector.y = 0.0
    grasp_msg.post_grasp_retreat.direction.vector.z = 1.0
    grasp_msg.post_grasp_retreat.min_distance = 0.05
    grasp_msg.post_grasp_retreat.desired_distance = 0.10
    
    # set the gripper as closed during grasping
    close_gripper(grasp_msg)

    # allow the contact with the object
    grasp_msg.allowed_touch_objects = ["object"]

    # for visualization and debug purposes
    rospy.sleep(0.05)
    viz_pose = PoseStamped()
    viz_pose.header.stamp = rospy.Time.now()
    viz_pose.header.frame_id = "map"
    viz_pose.pose = ee_pose
    rospy.sleep(0.05)
    nxt_EE_pose_pub.publish(viz_pose)
    
    # open the gripper before the execution of the pick action
    open_gripper(None)

    # try to pick the object
    rospy.loginfo("Trying to reach x:{}, y:{}, z:{}".format(
        ee_pose.position.x, ee_pose.position.y, ee_pose.position.z))
    arm_group.pick(object_name="object", grasp=grasp_msg)

    # rospy.sleep(0.05)
    
    arm_group.detach_object("object")
    
    # rospy.sleep(0.05)
    
    # remove the ogbject from the planning scene
    scene.remove_world_object(name="object")
    rospy.sleep(0.05)

    # plan to Insert_backpack pose
    arm_group.set_named_target("Insert_backpack")
    success = False
    tries = 0
    while success == False and tries < 3:
        rospy.loginfo("Trying to reach the backpack")
        rospy.sleep(0.05)
        success = arm_group.go(wait=True)
        tries = tries + 1
    arm_group.stop()

    open_gripper(None)
    
    rospy.sleep(0.05)
    
    reset_pan_tilt()
    
def add_new_pose(data):
    rospy.loginfo("received a new optimal base pose...")
    opt_base_poses.append(data)
    opt_base_poses_history.append(data)
    viz_all_base_poses()
    
def viz_all_base_poses():
    pose_array_msg = PoseArray()
    pose_array_msg.header.frame_id = "map"
    pose_array_msg.header.stamp = rospy.Time.now()
    
    for multi_target in opt_base_poses_history:
        pose_array_msg.poses.append(multi_target.base_pose.pose)
    
    rospy.sleep(0.01)
    all_base_poses_pub.publish(pose_array_msg)
    
    
    



# initialize the MoveIt stack
moveit_commander.roscpp_initialize(sys.argv)

# create a new ros node
rospy.init_node("move_to_next_pose")
rospy.loginfo("Created node 'move_to_next_pose'")

# publish an image with the ROI and the corrected grasp
img_pub = rospy.Publisher("/grasp_point_img", Image, queue_size=10)
rospy.loginfo("Created 'grasp_point_img' topic")

# publish a pose for the corrected grasp
pose_pub = rospy.Publisher("/grasp_corrected", PoseStamped, queue_size=10)
rospy.loginfo("Created 'grasp_corrected' topic")

# subscribe to the topic to command single joints
# joint_sing_comm = rospy.Publisher(
#     "/locobot/commands/joint_single", JointSingleCommand, queue_size=10)
pan_msg_pub = rospy.Publisher("/locobot/pan_controller/command", Float64, queue_size=10)
tilt_msg_pub = rospy.Publisher("/locobot/tilt_controller/command", Float64, queue_size=10)

# subscribe to a topic to receive the newly found pose for the mobile base
rospy.Subscriber("add_opt_base_pose", multi_target_pose, add_new_pose)
rospy.loginfo("Created topic 'add_opt_base_pose'")

# publish on topic to move the base
move_base_client = actionlib.SimpleActionClient("/locobot/move_base", MoveBaseAction)
move_base_client.wait_for_server()

nxt_EE_pose_pub = rospy.Publisher("/des_EE_pose", PoseStamped, queue_size=10)

# publish on a topic to visualize all the base poses
all_base_poses_pub = rospy.Publisher("/all_base_poses", PoseArray, queue_size=10)

# create a list of poses (queue)
opt_base_poses = []
opt_base_poses_history = []

gripper_pub = rospy.Publisher(
    "/locobot/gripper_controller/command", JointTrajectory, queue_size=10)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

arm_group = moveit_commander.MoveGroupCommander("interbotix_arm")
arm_group.set_max_velocity_scaling_factor(0.8)
arm_group.set_max_acceleration_scaling_factor(0.6)

gripper_group = moveit_commander.MoveGroupCommander("interbotix_gripper")
gripper_group.set_max_velocity_scaling_factor(0.8)
gripper_group.set_max_acceleration_scaling_factor(0.6)

# plan to Insert_backpack pose
arm_group.set_named_target("Insert_backpack")
success = False
tries = 0
while success == False and tries < 3:
    rospy.loginfo("Trying to reach the backpack")
    rospy.sleep(0.05)
    success = arm_group.go(wait=True)
    tries = tries + 1
arm_group.stop()


rospy.loginfo("Waiting for a new optimal base pose...")
# start looping on all the desired poses
while not rospy.is_shutdown():
    # check for empy
    if len(opt_base_poses) <= 0:
        rospy.sleep(0.05)
    else:
        # reset pan and tilt for navigation
        reset_pan_tilt()
        rospy.sleep(0.05)

        # extract the first element (oldest inserted)
        nxt_base_pose = opt_base_poses.pop(0)

        # create a MoveBaseGoal message
        goal_base_pose = MoveBaseGoal()

        goal_base_pose.target_pose.header.stamp = rospy.Time.now()
        goal_base_pose.target_pose.header.frame_id = 'map'

        goal_base_pose.target_pose.pose = nxt_base_pose.base_pose.pose

        # retry if fail
        while not rospy.is_shutdown():
            rospy.loginfo("Moving the base to the optimal pose...")
            move_base_client.send_goal(goal_base_pose)
            move_base_client.wait_for_result()
            res = move_base_client.get_result()
            state = move_base_client.get_state()

            if res is not None and state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Goal finished and succeeded.")
                break
            else:
                rospy.logwarn("Goal did not succeed, retrying...")

        # move the arm towards the desired targets
        for ee_pose_msg in nxt_base_pose.gripper_poses:
            pick_and_store(ee_pose_msg.pose, nxt_base_pose.base_yaw)
