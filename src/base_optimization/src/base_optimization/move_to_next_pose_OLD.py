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
from visualization_msgs.msg import Marker
from base_optimization.msg import multi_target_pose
from moveit_msgs.msg import MoveGroupAction, Grasp
from interbotix_xs_msgs.msg import JointSingleCommand
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from base_optimization.srv import images_giver, images_giverResponse
import pyrealsense2 as rs

ROI_SIZE = 150  # pixels

def set_pan_tilt_to_zero():
    # set tilt angle
    tilt_msg = JointSingleCommand()
    tilt_msg.name = 'tilt'
    tilt_msg.cmd = 0.0
    rospy.sleep(0.1)

    joint_sing_comm.publish(tilt_msg)

    # set pan angle
    pan_msg = JointSingleCommand()
    pan_msg.name = 'pan'
    pan_msg.cmd = 0.0
    rospy.sleep(0.1)

    joint_sing_comm.publish(pan_msg)

def reset_pan_tilt():
    # set tilt angle
    tilt_msg = JointSingleCommand()
    tilt_msg.name = 'tilt'
    tilt_msg.cmd = 0.2618
    rospy.sleep(0.1)

    joint_sing_comm.publish(tilt_msg)

    # set pan angle
    pan_msg = JointSingleCommand()
    pan_msg.name = 'pan'
    pan_msg.cmd = 0.0
    rospy.sleep(0.1)

    joint_sing_comm.publish(pan_msg)

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
    # coord_camera_norm = coord_camera/np.linalg.norm(coord_camera, ord=2)
    # roll = 0
    # pitch = -np.arctan2(coord_camera_norm[2],
    #                     np.sqrt(coord_camera_norm[0]**2 + coord_camera_norm[1]**2))
    # yaw = np.arctan2(coord_camera_norm[1], coord_camera_norm[0])

    # rospy.logwarn([pitch, yaw])
    
    # find yaw and pitch angle
    yaw = np.arctan2(coord_camera[1], coord_camera[0])
    pitch = np.arctan2(coord_camera[2], coord_camera[0])
    
    # set tilt angle
    tilt_msg = JointSingleCommand()
    tilt_msg.name = 'tilt'
    tilt_msg.cmd = -pitch
    rospy.sleep(0.1)

    joint_sing_comm.publish(tilt_msg)

    # set pan angle
    pan_msg = JointSingleCommand()
    pan_msg.name = 'pan'
    pan_msg.cmd = yaw
    rospy.sleep(0.1)

    joint_sing_comm.publish(pan_msg)
    
    return pitch, yaw

def get_latest_images():
    bridge = CvBridge()
    rospy.wait_for_service('images_giver_service')
    try:
        proxy = rospy.ServiceProxy('images_giver_service', images_giver)
        resp = proxy()
        rgb_img = bridge.imgmsg_to_cv2(resp.color_image, 'bgr8')
        depth_img = bridge.imgmsg_to_cv2(resp.depth_image, desired_encoding='passthrough')
        cam_info = resp.camera_info
        return rgb_img, depth_img, cam_info
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return None, None, None

def publish_marker(marker_pub, position, frame_id, marker_id=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "clicked_points"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)

def mouse_callback(event, x, y, flags, param):
    latest_rgb, latest_depth, latest_cam_info = param['latest_data']
    last_clicked_coords = param['last_clicked_coords']
    tf_buffer = param['tf_buffer']
    marker_pub = param['marker_pub']
    target_frame = param['target_frame']

    if event == cv2.EVENT_LBUTTONDOWN and latest_depth is not None and latest_cam_info is not None:
        depth = latest_depth[y, x].astype(np.float32) / 1000.0  # depth in meters
        if depth == 0 or np.isnan(depth):
            print("No valid depth at clicked point.")
            return

        K = np.array(latest_cam_info.K).reshape(3,3)
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth + 0.01

        print(f"Clicked pixel: ({x}, {y}) → Camera frame: ({X:.3f}, {Y:.3f}, {Z:.3f}) meters")
        camera_frame = latest_cam_info.header.frame_id

        point_cam = PointStamped()
        point_cam.header.stamp = rospy.Time.now()
        point_cam.header.frame_id = camera_frame
        point_cam.point.x = X
        point_cam.point.y = Y
        point_cam.point.z = Z

        try:
            point_map = tf_buffer.transform(point_cam, target_frame, timeout=rospy.Duration(1.0))
            xyz = [point_map.point.x, point_map.point.y, point_map.point.z]
            print(f"Transformed to '{target_frame}': ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) meters")
            publish_marker(marker_pub, xyz, target_frame)
            last_clicked_coords[0] = xyz  # Save last clicked fixed-frame coordinates
        except Exception as e:
            print(f"TF transform failed: {e}")

def correct_ee_pose(ee_pose_original):
    rospy.sleep(0.5)
    set_pan_tilt_to_zero()

    # set pan and tilt in a way to have the center
    # of the image aligned to original grasping pose
    rospy.sleep(1)
    pan, tilt = set_pan_tilt(ee_pose_original)

    rospy.sleep(5)
    bridge = CvBridge()
    rospy.wait_for_service("/locobot/images_giver_service")
    get_igms = rospy.ServiceProxy("/locobot/images_giver_service", images_giver)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    print(f"Click a point in the RGB image window to get its 3D coordinates in the map frame and visualize in RViz.")

    cv2.namedWindow('RGB Image')
    latest_data = [None, None, None]
    last_clicked_coords = [None]  # Store last clicked fixed-frame coordinates

    cv2.setMouseCallback('RGB Image', mouse_callback, {
        'latest_data': latest_data,
        'last_clicked_coords': last_clicked_coords,
        'tf_buffer': tf_buffer,
        'marker_pub': marker_pub,
        'target_frame': 'map'
    })

    while cv2.getWindowProperty('RGB Image', cv2.WND_PROP_VISIBLE) >= 1 and not rospy.is_shutdown():
        rgb_img, depth_img, cam_info = get_latest_images()
        if rgb_img is not None:
            latest_data[0] = rgb_img
            latest_data[1] = depth_img
            latest_data[2] = cam_info
            cv2.imshow('RGB Image', rgb_img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

    # Return or print the last clicked coordinates in the fixed frame
    if last_clicked_coords[0] is not None:
        print(f"\nLast clicked point in map frame: {last_clicked_coords[0]}") 
    else:
        print("\nNo point was clicked.")
    
    reset_pan_tilt()
    
    return pan, tilt, last_clicked_coords[0]
    
def open_gripper(grasp_msg=None):
    global gripper_pub

    point = JointTrajectoryPoint()
    point.positions = [0.5]
    point.velocities = [1]
    point.accelerations = [0.5]
    point.time_from_start.secs = 1

    joint_trajectory = JointTrajectory()
    joint_trajectory.joint_names = ["left_finger"]
    # joint_trajectory.joint_names.clear()
    joint_trajectory.points.append(point)

    if grasp_msg is not None:
        grasp_msg.pre_grasp_posture = joint_trajectory
    else:
        rospy.sleep(1)
        gripper_pub.publish(joint_trajectory)
        rospy.sleep(2)

def close_gripper(grasp_msg=None):
    global gripper_pub

    point = JointTrajectoryPoint()
    point.positions = [-0.3]
    point.velocities = [1]
    point.accelerations = [0.5]
    point.time_from_start.secs = 1

    joint_trajectory = JointTrajectory()
    joint_trajectory.joint_names = ["left_finger"]
    # joint_trajectory.joint_names.clear()
    joint_trajectory.points.append(point)

    if grasp_msg is not None:
        grasp_msg.grasp_posture = joint_trajectory
    else:
        rospy.sleep(1)
        gripper_pub.publish(joint_trajectory)
        rospy.sleep(2)

def pick_and_store(ee_pose, yaw_angle):
    pick_success = -1
    while pick_success == -1:
        # refine the ee_grasp basing on the camera input
        pan, tilt, coords = correct_ee_pose(ee_pose)

        # add a collision object (box)
        object_pose = PoseStamped()
        object_pose.header.frame_id = "map"
        object_pose.pose.position.x = coords[0]
        object_pose.pose.position.y = coords[1]
        object_pose.pose.position.z = coords[2]
        object_pose.pose.orientation.x = 0.0
        object_pose.pose.orientation.y = 0.0
        object_pose.pose.orientation.z = 0.0
        object_pose.pose.orientation.w = 1.0

        scene.add_box(name="object", pose=object_pose, size=(0.03, 0.03, 0.03))
        rospy.sleep(1)

        # compute the grap pose orientation
        roll = 0
        pitch = 0
        yaw = yaw_angle
        # rospy.logwarn(yaw)

        quat = tf.transformations.quaternion_from_euler(
            roll, pitch, yaw, axes='sxyz')

        # create a Grasp pipeline message
        grasp_msg = Grasp()

        # set the grasp pose
        grasp_msg.grasp_pose.header.frame_id = "map"
        grasp_msg.grasp_pose.pose.position.x = coords[0]
        grasp_msg.grasp_pose.pose.position.y = coords[1]
        grasp_msg.grasp_pose.pose.position.z = coords[2]
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

        # allow the contact with the object
        grasp_msg.allowed_touch_objects = ["object"]

        # set the gripper fingers as open before grasping
        # open_gripper(grasp_msg)
        open_gripper(None)

        # set gripper as closed during grasping
        # close_gripper(grasp_msg)

        # plan to a gripper pose
        # arm_group.set_position_target([ee_pose.position.x, ee_pose.position.y, ee_pose.position.z])
        # ee_pose.orientation.x = quat[0]
        # ee_pose.orientation.y = quat[1]
        # ee_pose.orientation.z = quat[2]
        # ee_pose.orientation.w = quat[3]
        # for visualization and debug purposes
        rospy.sleep(0.1)
        viz_pose = PoseStamped()
        viz_pose.header.stamp = rospy.Time.now()
        viz_pose.header.frame_id = "map"
        viz_pose.pose.position.x = coords[0]
        viz_pose.pose.position.y = coords[1]
        viz_pose.pose.position.z = coords[2]
        viz_pose.pose.orientation.x = quat[0] 
        viz_pose.pose.orientation.y = quat[1]
        viz_pose.pose.orientation.z = quat[2]
        viz_pose.pose.orientation.w = quat[3]
        rospy.sleep(0.1)
        nxt_EE_pose_pub.publish(viz_pose)


        # open_gripper()

        # try to pick the object
        rospy.loginfo("Trying to reach x:{}, y:{}, z:{}".format(
            coords[0], coords[1], coords[2]))
        pick_success = arm_group.pick(object_name="object", grasp=grasp_msg)
        print(pick_success)

    
    # arm_group.set_pose_target(ee_pose)
    # arm_group.set_pose_reference_frame("map")
    # success = False
    # tries = 0
    # while success==False and tries<3:
    #     rospy.loginfo("Trying to reach x:{}, y:{}, z:{}".format(ee_pose.position.x, ee_pose.position.y, ee_pose.position.z))
    #     rospy.sleep(1)
    #     success = arm_group.go(wait=True)
    #     tries = tries + 1
    # arm_group.stop()
    # arm_group.clear_pose_targets()

    rospy.sleep(1)

    close_gripper(None)
    
    rospy.sleep(1)
    
    arm_group.detach_object("object")
    rospy.sleep(1)
    # remove the ogbject from the planning scene
    scene.remove_world_object(name="object")
    rospy.sleep(1)

    # plan to Insert_backpack pose
    arm_group.set_named_target("Insert_backpack")
    success = False
    tries = 0
    while success == False and tries < 3:
        rospy.loginfo("Trying to reach the backpack")
        rospy.sleep(1)
        success = arm_group.go(wait=True)
        tries = tries + 1
    arm_group.stop()

    open_gripper(None)
    



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
joint_sing_comm = rospy.Publisher(
    "/locobot/commands/joint_single", JointSingleCommand, queue_size=10)

# subscribe to a topic to receive the newly found pose for the mobile base
rospy.Subscriber("add_opt_base_pose", multi_target_pose, add_new_pose)
rospy.loginfo("Created topic 'add_opt_base_pose'")

# publish on topic to move the base
move_base_client = actionlib.SimpleActionClient(
    "/locobot/move_base", MoveBaseAction)
move_base_client.wait_for_server()

nxt_EE_pose_pub = rospy.Publisher("/des_EE_pose", PoseStamped, queue_size=10)

# publish on a topic to visualize all the base poses
all_base_poses_pub = rospy.Publisher("/all_base_poses", PoseArray, queue_size=10)

marker_pub = rospy.Publisher('clicked_point_marker', Marker, queue_size=10)

# create a list of poses (queue)
opt_base_poses = []
opt_base_poses_history = []

gripper_pub = rospy.Publisher(
    "/locobot/gripper_controller/command", JointTrajectory, queue_size=10)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

arm_group = moveit_commander.MoveGroupCommander("interbotix_arm")
arm_group.set_max_velocity_scaling_factor(0.8)
arm_group.set_max_acceleration_scaling_factor(0.8)

gripper_group = moveit_commander.MoveGroupCommander("interbotix_gripper")
gripper_group.set_max_velocity_scaling_factor(1)
gripper_group.set_max_acceleration_scaling_factor(0.8)

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
        # reset pan and tilt for navigation
        reset_pan_tilt()
        rospy.sleep(2)

        # extract the first element (oldest inserted)
        nxt_base_pose = opt_base_poses.pop(0)

        # create a MoveBaseGoal message
        goal_base_pose = MoveBaseGoal()

        goal_base_pose.target_pose.header.stamp = rospy.Time.now()
        goal_base_pose.target_pose.header.frame_id = 'map'

        goal_base_pose.target_pose.pose = nxt_base_pose.base_pose.pose

        # retry if fail
        # while not rospy.is_shutdown():
        #     rospy.loginfo("Moving the base to the optimal pose...")
        #     move_base_client.send_goal(goal_base_pose)
        #     move_base_client.wait_for_result()
        #     res = move_base_client.get_result()
        #     state = move_base_client.get_state()

        #     if res is not None and state == actionlib.GoalStatus.SUCCEEDED:
        #         rospy.loginfo("Goal finished and succeeded.")
        #         break
        #     else:
        #         rospy.logwarn("Goal did not succeed, retrying...")

        # # move the arm towards the desired targets
        # for ee_pose_msg in nxt_base_pose.gripper_poses:
        #     pick_and_store(ee_pose_msg.pose, nxt_base_pose.base_yaw)
