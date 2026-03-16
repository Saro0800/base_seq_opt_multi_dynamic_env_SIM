import octomap_msgs.msg
import rospy
import tf.transformations
import tf2_ros
import tf
import tf2_geometry_msgs
import tf
import numpy as np
import ros_numpy
import cv2
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from scipy.spatial.transform import Rotation

from reach_space_modeling.srv import ell_params, ell_paramsResponse
from base_optimization.problem_formulation_collision_multi import BasePoseOptProblem

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.termination.collection import TerminationCollection
from pymoo.core.termination import Termination

from base_optimization.srv import octomap2cloud, octomap2cloudResponse
from base_optimization.msg import multi_target_pose
from std_msgs.msg import Bool

import networkx as nx   
from networkx.algorithms.approximation import traveling_salesman_problem
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import DBSCAN


def send_opt_base_pose(base_pose, reachable_EE_position, reachable_EE_orientation):
    # create a PoseStamped msg for the base pose
    base_pose_msg = PoseStamped()
    base_pose_msg.header.stamp = rospy.Time.now()
    base_pose_msg.header.frame_id = 'octomap_frame'

    base_pose_msg.pose.position.x = base_pose[0]
    base_pose_msg.pose.position.y = base_pose[1]
    base_pose_msg.pose.position.z = 0

    quat = tf.transformations.quaternion_from_euler(
        0, 0, np.deg2rad(base_pose[2]), axes='sxyz')
    base_pose_msg.pose.orientation.x = quat[0]
    base_pose_msg.pose.orientation.y = quat[1]
    base_pose_msg.pose.orientation.z = quat[2]
    base_pose_msg.pose.orientation.w = quat[3]
    
    # create a list of PoseStamped msgs representing the EE targets
    gripper_poses_list = []
    for i, EE_pose in enumerate(reachable_EE_position):
        EE_pose_msg = PoseStamped()
        EE_pose_msg.header.stamp = rospy.Time.now()
        EE_pose_msg.header.frame_id = 'octomap_frame'
        
        EE_pose_msg.pose.position.x = EE_pose[0]
        EE_pose_msg.pose.position.y = EE_pose[1]
        EE_pose_msg.pose.position.z = EE_pose[2]
        
        EE_pose_msg.pose.orientation.x = reachable_EE_orientation[i][0]
        EE_pose_msg.pose.orientation.y = reachable_EE_orientation[i][1]
        EE_pose_msg.pose.orientation.z = reachable_EE_orientation[i][2]
        EE_pose_msg.pose.orientation.w = reachable_EE_orientation[i][3]
        
        gripper_poses_list.append(EE_pose_msg)

    # # publish the goal message
    # rospy.loginfo("Sending a new optimal base pose...")
    # next_pose_topic.publish(base_pose)
    
    # create the custom multi_target_pose msg
    multi_target_pose_msg = multi_target_pose()
    multi_target_pose_msg.base_yaw = np.deg2rad(base_pose[2])
    multi_target_pose_msg.base_pose = base_pose_msg
    multi_target_pose_msg.gripper_poses = gripper_poses_list
    
    next_pose_topic.publish(multi_target_pose_msg)

def find_opt_base_pose(ell_frame_link, des_position_multi, des_orientation_multi, point_cloud,
                       free_space_2d, weights, occupancy_grid_info):
    
    # Lists to accumulate all base poses and corresponding EE poses
    all_base_poses = []
    all_reachable_EE_positions = []
    all_reachable_EE_orientations = []
    optimization_success = False
    
    while len(des_position_multi) > 0:
        rospy.loginfo("Looking for an optimal base pose...")
        opt_start_time = rospy.Time.now()
        # define the optimization problem to find the optimal base pose

        problem = BasePoseOptProblem(ell_center_map,
                                     ell_axis_out*0.95,
                                     ell_axis_inn,
                                     des_position_multi,
                                     weights,
                                     point_cloud,
                                     free_space_2d,
                                     occupancy_grid_info,
                                     ell_center_base)

        # solve the optimization problem using the PSO algorithm
        algorithm = PSO()
        # algorithm = GA()
        # algorithm = NSGA2()

        # define termination condition
        # - maximum number of iteration
        max_gen_termination = get_termination("n_gen", 5000)
        # - maximum time
        time_termination = get_termination("time", "00:01:00")
        # - tolerance on the objective function
        ftol_termination = RobustTermination(
            SingleObjectiveSpaceTermination(tol=pow(10, -3))
            )
        
        # combine termination criteria with OR logic (stops when any is satisfied)
        termination = TerminationCollection(max_gen_termination,
                                            time_termination,
                                            ftol_termination
                                            )

        # solve the optimization problem
        res = minimize(problem=problem,
                       algorithm=algorithm,
                       termination=termination,
                       verbose=False,
                       seed=1)

        
        # check which termination criterion was satisfied
        # after minimize(), termination state is in res.algorithm.termination,
        # not in the original termination objects passed to minimize()
        final_termination = res.algorithm.termination
        converged = False
        
        # access the individual terminations from the collection
        # order: [0]=max_gen, [1]=time, [2]=ftol
        if final_termination.terminations[2].perc >= 1.0:
            # tolerance on objective function was reached -> converged
            rospy.logwarn("Optimization converged: tolerance on objective function reached")
            converged = True
        elif final_termination.terminations[1].perc >= 1.0:
            # time limit was reached
            rospy.logwarn("Optimization terminated: time limit reached")
            break
        elif final_termination.terminations[0].perc >= 1.0:
            # maximum number of iterations was reached
            rospy.logwarn("Optimization terminated: maximum number of iterations reached")
            break
        else:
            rospy.logwarn("Optimization terminated for unknown reason")
            break

        if converged==True:
            opt_elapsed = (rospy.Time.now() - opt_start_time).to_sec()
            rospy.logwarn("Optimization time for this base pose: %.3f seconds", opt_elapsed)
            rospy.loginfo("Optimal base pose: x=%.4f, y=%.4f, theta=%.4f",
                          res.X[0], res.X[1], res.X[2])
            print(res.F)
        else:
            rospy.logwarn("No solution found for the desired targets! Stopping")
            break
        
        # obtain the ellipsoid position w.r.t. R0
        # find the base coordinates that place the ellipsoid center
        # at the desired position

        homog_matr = np.zeros((4, 4))
        homog_matr[:3, :3] = Rotation.from_euler(
            'xyz', [0, 0, res.X[2]], degrees=True).as_matrix()
        homog_matr[:3, 3] = np.array([res.X[0], res.X[1], 0])
        homog_matr[3, 3] = 1
        base_pos = np.dot(homog_matr, np.array(
            [-ell_center_base[0], -ell_center_base[1], 0, 1]))
        base_pos[2] = res.X[2]
        
        
        still_to_reach = []
        reachable_EE_position = []
        reachable_EE_orientation = []
        for i, p in enumerate(des_position_multi):
            # if inside the outer and outside the inner
            if ((res.X[0]-p.x)/ell_axis_out[0])**2 + ((res.X[1]-p.y)/ell_axis_out[1])**2 + ((ell_center_map[2]-p.z)/ell_axis_out[2])**2 <= 1 and\
                ((res.X[0]-p.x)/ell_axis_inn[0])**2 + ((res.X[1]-p.y)/ell_axis_inn[1])**2 + ((ell_center_map[2]-p.z)/ell_axis_inn[2])**2 > 1:
                
                # create a list of EE targets reachable from the base pose found
                print("point ({:.2f}, {:.2f}, {:.2f}) is reachable".format(p.x, p.y, p.z))
                reachable_EE_position.append([p.x, p.y, p.z])
                quat = [des_orientation_multi[i][0], des_orientation_multi[i][1], des_orientation_multi[i][2], des_orientation_multi[i][3]]
                reachable_EE_orientation.append(quat)
            else:
                still_to_reach.append(p)
        
        # if optimization did not converge and no points are reachable,
        # the remaining targets are assumend to be not reachable
        if not converged and len(reachable_EE_position) == 0:
            rospy.logwarn("\u2717 Optimization did not converge and no points are reachable. Aborting.")
            break
        
        # append the base pose found the rest of the sequence
        all_base_poses.append([base_pos[0], base_pos[1], base_pos[2]])
        
        # append the list of reachable targets
        all_reachable_EE_positions.append(reachable_EE_position)
        all_reachable_EE_orientations.append(reachable_EE_orientation)
        optimization_success = True
        
        des_position_multi = still_to_reach

    # send all base poses
    if optimization_success:
        rospy.logwarn("Sending all %d optimal base poses...", len(all_base_poses))
        
        # send each base poses and associated targets separatly
        for i in range(len(all_base_poses)):
            send_opt_base_pose(all_base_poses[i], all_reachable_EE_positions[i], all_reachable_EE_orientations[i])
    else:
        rospy.logwarn("No valid base poses found. Nothing to send.")
    
    # Publish optimization status (True = success, False = failure)
    opt_status_pub.publish(Bool(data=optimization_success))
 
def compute_free_2d_space():
    # retrieve the projected 2D occupancy grid map
    msg = rospy.wait_for_message('/locobot/octomap_server/projected_map', OccupancyGrid)

    # convert OccupancyGrid to OpenCV image
    width = msg.info.width
    height = msg.info.height
    
    # convert data to numpy array
    grid_data = np.array(msg.data, dtype=np.uint8).reshape((height, width))

    # values in grid_data are probabilities from 0 to 100, with -1 for unknown
    # - cells with value 0 are free
    # - cells with value 100 are occupied
    # - cells with value -1 are unknown

    # convert unknown cells to occupied
    image = np.where(grid_data == -1, 100, grid_data)
    
    # convert to a grayscale image 
    # black (0) = occupied, white (255) = free
    image = np.uint8(255 - image * 2.55)
    
    # flip the image vertically to correct ROS to OpenCV coordinate system
    # ROS: origin at bottom-left, Y-axis up
    # OpenCV: origin at top-left, Y-axis down
    image = cv2.flip(image, 0)
    
    # cv2.imshow('Original Occupancy Grid', image)
    # cv2.waitKey(1)
    
    # cv2.imwrite("/home/humans/base_seq_opt_multi_dynamic_env_SIM/src/base_optimization/img/occupancy_grid.png",
    #             image)

    # inflate obstacles with radius d (in meters)
    robot_rad = 0.175
    obstacle_dist = 0.05
    d_m = robot_rad + obstacle_dist
    resolution = msg.info.resolution
    
    # inflation radius in pixels
    d = int(d_m / resolution)

    # create the inflation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d+1, 2*d+1))

    # create binary mask where obstacles are white (occupied cells have low values)
    obstacle_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)[1]
    
    # cv2.imshow('Obstacle Mask', obstacle_mask)
    # cv2.waitKey(1)

    # dilate obstacles
    inflated_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
    
    # create an image with all obstacels inflated
    # free space is visible as white
    image_inflated = image
    image_inflated[inflated_mask > 0] = 0

    # cv2.imshow('Inflated Occupancy Grid', image_inflated)
    # cv2.waitKey(1000)
    
    # cv2.imwrite("/home/humans/base_seq_opt_multi_dynamic_env_SIM/src/base_optimization/img/inflated_grid.png",
    #             image_inflated)

    # keep only the connected white space containing the robot's position
    # retrieve robot position
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_buffer.can_transform("octomap_frame", "locobot/base_footprint", rospy.Time(0))
    transform = tf_buffer.lookup_transform("octomap_frame", "locobot/base_footprint", rospy.Time(0), rospy.Duration(1))
    
    robot_x = transform.transform.translation.x
    robot_y = transform.transform.translation.y
    
    # convert robot position from map frame to pixel coordinates
    origin_x = msg.info.origin.position.x
    origin_y = msg.info.origin.position.y
    robot_pixel_x = int((robot_x - origin_x) / resolution)
    robot_pixel_y = int((robot_y - origin_y) / resolution)
    
    # flip y coordinate because we flipped the image
    robot_pixel_y = height - 1 - robot_pixel_y
        
    # flood fill from robot position
    free_space = image_inflated.copy()
    cv2.floodFill(free_space, None, (robot_pixel_x, robot_pixel_y), 128)
        
    # Keep only the flooded region (value 128), set everything else to black (0)
    image_inflated_robot_space = np.where(free_space == 128, 255, 0).astype(np.uint8)
    # cv2.circle(image_inflated_robot_space, (robot_pixel_x, robot_pixel_y), 3, (190), -1)
    
    # cv2.imshow('Robot Free Space', image_inflated_robot_space)
    # cv2.waitKey(0)
    # cv2.startWindowThread()

    # cv2.imwrite("/home/humans/base_seq_opt_multi_dynamic_env_SIM/src/base_optimization/img/robot_free_space.png",
    #             image_inflated_robot_space)

    return msg.info, image_inflated_robot_space

def handle_des_EE_pose_multi(pose_array_msg):
    rospy.logwarn("Received")
    # retrieve the set of desired poses
    des_position_multi = []
    des_orientation_multi = []
    for pose in pose_array_msg.poses:
        point = Point()
        point.x = pose.position.x
        point.y = pose.position.y
        point.z = pose.position.z
        des_position_multi.append(point)
        
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        des_orientation_multi.append(quat)
    
    weights = np.ones(len(des_position_multi))
    
    # compute the free 2D space where the mobile base can move
    occupancy_grid_info, free_space_2d = compute_free_2d_space()

    # request the conversion from octomap to point cloud
    rospy.loginfo("Waiting for service '/locobot/octomap2cloud_converter_srv' ...")
    rospy.wait_for_service("/locobot/octomap2cloud_converter_srv")
    octomap2cloud_srv = rospy.ServiceProxy("/locobot/octomap2cloud_converter_srv", octomap2cloud)

    rospy.loginfo("Sending a service request to '/locobot/octomap2cloud_converter_srv'...")
    occ_cloud = octomap2cloud_srv()
    rospy.loginfo("Response received")

    # convert the PointCloud2 message into a numpy array
    cloud_np = ros_numpy.numpify(occ_cloud.cloud)

    # find the optimal base pose
    find_opt_base_pose(ell_ref_frame, des_position_multi, des_orientation_multi,
                       cloud_np, free_space_2d, weights, occupancy_grid_info)


# create a ROS node
rospy.init_node('find_opt_pose')

# retrieve the parameter of the ellipsoid
rospy.loginfo("Waiting for serive /get_ellipsoid_params...")
rospy.wait_for_service('get_ellipsoid_params')
ell_params_srv = rospy.ServiceProxy('get_ellipsoid_params', ell_params)
rospy.loginfo("Service /get_ellipsoid_params is available")

rospy.loginfo("Sending request to /get_ellipsoid_params...")
ell_par = ell_params_srv()
rospy.loginfo("Ellipsoid parameters received:\n\
                \txC=%.4f, yC=%.4f, zC=%.4f\n\
                \taO=%.4f, bO=%.4f, cO=%.4f,\n\
                \taI=%.4f, bI=%.4f, cI=%.4f",
              ell_par.xC, ell_par.yC, ell_par.zC,
              ell_par.aO, ell_par.bO, ell_par.cO,
              ell_par.aI, ell_par.bI, ell_par.cI)
rospy.loginfo("Ellipsoid reference frame: %s", ell_par.ell_ref_frame)

ell_center = np.array([ell_par.xC, ell_par.yC, ell_par.zC])
ell_axis_out = np.array([ell_par.aO, ell_par.bO, ell_par.cO])
ell_axis_inn = np.array([ell_par.aI, ell_par.bI, ell_par.cI])
ell_ref_frame = ell_par.ell_ref_frame

# transform the center from ell_ref_fram to the fixed frame ("map")
tmp_pose = PoseStamped()
tmp_pose.header.stamp = rospy.Time.now()
tmp_pose.header.frame_id = ell_ref_frame

tmp_pose.pose.position.x = ell_center[0]
tmp_pose.pose.position.y = ell_center[1]
tmp_pose.pose.position.z = ell_center[2]

tmp_pose.pose.orientation.x = 0.0
tmp_pose.pose.orientation.y = 0.0
tmp_pose.pose.orientation.z = 0.0
tmp_pose.pose.orientation.w = 1.0

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

tf_buffer.can_transform("octomap_frame", ell_ref_frame, rospy.Time(0))
transform = tf_buffer.lookup_transform(
    "octomap_frame", ell_ref_frame, rospy.Time(0), rospy.Duration(1))
ell_transformed_msg = tf2_geometry_msgs.do_transform_pose(tmp_pose, transform)

ell_center_map = np.array([ell_transformed_msg.pose.position.x,
                           ell_transformed_msg.pose.position.y,
                           ell_transformed_msg.pose.position.z])

# the reference frame attacched to the center of the ellipsoid
# is fixed with respect to the reference frame of the mobile base
# for what concern both the orientation and the position.
# here the translation vector between these two ref. frames is built
tf_buffer.can_transform("locobot/base_footprint", ell_ref_frame, rospy.Time(0))
transform = tf_buffer.lookup_transform(
    "locobot/base_footprint", ell_ref_frame, rospy.Time(0), rospy.Duration(1))
ell_transformed_msg = tf2_geometry_msgs.do_transform_pose(tmp_pose, transform)
ell_center_base = np.array([ell_transformed_msg.pose.position.x,
                            ell_transformed_msg.pose.position.y,
                            ell_transformed_msg.pose.position.z])

rospy.Subscriber("/gripper_poses", PoseArray, callback=handle_des_EE_pose_multi)
tmp_pub = rospy.Publisher("/des_EE_pose_tmp", PoseStamped, queue_size=10)

rospy.sleep(0.5)

# publish on topic which add a new optimal base pose to a queue
next_pose_topic = rospy.Publisher("/locobot/add_opt_base_pose", multi_target_pose, queue_size=100)

# publish optimization status (success/failure) after each optimization run
opt_status_pub = rospy.Publisher("/optimization_status", Bool, queue_size=10)

print()
rospy.loginfo("Waiting for the desired end-effector pose...")
rospy.spin()
