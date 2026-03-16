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
        
        # Combine termination criteria with OR logic (stops when any is satisfied)
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

        
        # Check which termination criterion was satisfied
        # NOTE: After minimize(), termination state is in res.algorithm.termination,
        # not in the original termination objects passed to minimize()
        final_termination = res.algorithm.termination
        converged = False
        
        # Access the individual terminations from the collection
        # Order: [0]=max_gen, [1]=time, [2]=ftol
        if final_termination.terminations[2].perc >= 1.0:
            # Tolerance on objective function was reached → success
            rospy.loginfo("\u2713 Optimization converged: tolerance on objective function reached")
            converged = True
        elif final_termination.terminations[1].perc >= 1.0:
            # Time limit was reached
            rospy.logwarn("\u2717 Optimization terminated: time limit reached")
            break
        elif final_termination.terminations[0].perc >= 1.0:
            # Maximum number of iterations was reached
            rospy.logwarn("\u2717 Optimization terminated: maximum number of iterations reached")
            break
        else:
            rospy.logwarn("\u2717 Optimization terminated for unknown reason")
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
        
        # Obtain the ellipsoid position w.r.t. R0
        # Find the base coordinates that place the ellipsoid center
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
                print("point ({:.2f}, {:.2f}, {:.2f}) is reachable".format(p.x, p.y, p.z))
                reachable_EE_position.append([p.x, p.y, p.z])
                quat = [des_orientation_multi[i][0], des_orientation_multi[i][1], des_orientation_multi[i][2], des_orientation_multi[i][3]]
                reachable_EE_orientation.append(quat)
            else:
                still_to_reach.append(p)
        
        # If optimization did not converge and no points are reachable,
        # the result is not useful: stop and signal failure
        if not converged and len(reachable_EE_position) == 0:
            rospy.logwarn("\u2717 Optimization did not converge and no points are reachable. Aborting.")
            break
        
        all_base_poses.append([base_pos[0], base_pos[1], base_pos[2]])
        all_reachable_EE_positions.append(reachable_EE_position)
        all_reachable_EE_orientations.append(reachable_EE_orientation)
        optimization_success = True
        
        des_position_multi = still_to_reach

    # send all base poses
    if optimization_success:
        rospy.loginfo("\u2713 Sending all %d optimal base poses...", len(all_base_poses))
        for i in range(len(all_base_poses)):
            send_opt_base_pose(all_base_poses[i], all_reachable_EE_positions[i], all_reachable_EE_orientations[i])
    else:
        rospy.logwarn("\u2717 No valid base poses found. Nothing to send.")
    
    # Publish optimization status (True = success, False = failure)
    opt_status_pub.publish(Bool(data=optimization_success))

def find_hamiltonian_path_greedy(g: nx.Graph):
    # since we want to traverse all nodes and the graph is
    # fully connected, the total path length is equal to the
    # number of nodes in the graph
    path = []
    
    # set the first and the last node of the path
    path.append('robot')
    
    # set first dummy sequence of nodes
    for n in range(len(g.nodes)-2):
        path.append(0)
    
    # set the last node of the path
    path.append('depot')
        
    # nodes associated to the poses are inserted from 2 directions in the
    # same iteration:
    #   1- starting from the closest to the 'robot' node and going forward
    #   2- starting from the closest to the 'depot' node and going backward
    head = 'robot'
    tail = 'depot'
    
    # flag array to check if a node has been inserted already
    # is_present[i] correspondes to the node named 'i'
    # there is no need for a flag associated to 'robot' and 'depot'
    # since they are already in the path 
    is_traversed = [False] * (len(g.nodes)-2)
    
    # count now many nodes have been traversed from head and tail
    nodes_from_head = 0
    nodes_from_tail = 0
    
    while (nodes_from_head + nodes_from_tail) < len(g.nodes)-2:
        # ---------------- ADD CLOSEST TO HEAD ----------------
        # create a dictionary where the key is the edge and the value is the weight
        # excluding self-loop
        head_dict = {}
        for n in g.nodes:
            if n==head:
                continue
            head_dict[(head, n)] = g.edges[(head, n)]['weight']
        
        # order the dictionary by value, hence, order the edges dictionary
        # in ascending order of edge weight
        head_dict_sorted = dict(sorted(head_dict.items(), key=lambda item: item[1]))
        
        # scan the sorted dict in ascending order of values
        # to add the node closest to head that has not been traverse yet
        for key, val in head_dict_sorted.items():
            # print("closest node to head ({:}) is: {:}".format(head, key[1]))
            # consider adding the neighbour of head as following node
            # in path
            if key[1] == 'robot':
                continue
            
            # this if is true if the closest node is 'depot' but
            # there are still nodes to insert in path
            if key[1] == 'depot':
                continue
            
            # check if the closest node has been traversed already
            if is_traversed[int(key[1])] == True:
                # print("Already traversed.")
                continue
            
            # add the neighbour as following node in path
            nodes_from_head = nodes_from_head + 1
            path[nodes_from_head] = key[1]
            # print("Added {:} because closest to head ({:}): {:}".format(key[1], head, val))
            
            # update is_traversed
            is_traversed[int(key[1])] = True
            
            # updated head
            head = key[1]
            
            break
        
        # check if all have been inserted
        if (nodes_from_head + nodes_from_tail) >= len(g.nodes)-2:
            break
        
        # ---------------- ADD CLOSEST TO TAIL ----------------
        # create a dictionary where the key is the edge and the value is the weight
        # excluding self-loop
        tail_dict = {}
        for n in g.nodes:
            if n==tail:
                continue
            tail_dict[(tail, n)] = g.edges[(tail, n)]['weight']
        
        # order the dictionary by value, hence, order the edges dictionary
        # in ascending order of edge weight
        tail_dict_sorted = dict(sorted(tail_dict.items(), key=lambda item: item[1]))
        
        # scan the sorted dict in ascending order of values
        # to add the node closest to tail that has not been traverse yet
        for key, val in tail_dict_sorted.items():
            # print("closest node to tail ({:}) is: {:}".format(tail, key[1]))
            # consider adding the neighbour of tail as following node
            # in path
            if key[1] == 'depot':
                continue
            
            # this if is true if the closest node is 'robot' but
            # there are still nodes to insert in path
            if key[1] == 'robot':
                continue
            
            # check if the closest node has been traversed already
            if is_traversed[int(key[1])] == True:
                # print("Already traversed")
                continue
            
            # add the neighbour as following node in path
            nodes_from_tail = nodes_from_tail + 1
            path[len(g.nodes)-1-nodes_from_tail] = key[1]
            # print("Added {:} because closest to tail ({:}): {:}".format(key[1], tail, val))
            
            # update is_traversed
            is_traversed[int(key[1])] = True
            
            # updated head
            tail = key[1]
            
            break
    
    return path
    
def compute_weights_clustering(des_position_multi):
    # define a the desired points
    des_points = []
    for pose in des_position_multi:
        des_points.append([pose.x, pose.y, pose.z])
        
    # apply clustering
    dbscan = DBSCAN(eps=0.45, min_samples=1)
    labels = dbscan.fit_predict(np.array(des_points))

    # create a dict where:
    #   - keys are cluster indeces
    #   - values are poses in the cluster
    cluster_dict = {}
    for l in labels:
        cluster_dict[l] = []

    for i, l in enumerate(labels):
        cluster_dict[l].append(des_points[i])

    # create the graph
    g = nx.Graph()

    # add the clusters as nodes
    for n in cluster_dict.keys():
        g.add_node(n)
        
        # add the poses in the cluster as node attribute
        g.nodes[n]['poses'] = cluster_dict[n]
        
        # compute the centroid
        centroid = np.mean(cluster_dict[n], axis=0)
        g.nodes[n]['centroid'] = centroid

    # add 'robot' and 'depot' nodes
    g.add_node('robot')
    g.nodes['robot']['poses'] = [0.0, 0.0, 0.0]
    g.nodes['robot']['centroid'] = [0.0, 0.0, 0.0]

    g.add_node('depot')
    g.nodes['depot']['poses'] = [6.0, -0.32, 0.0]
    g.nodes['depot']['centroid'] = [6.0, -0.32, 0.0]


    # add all edges to create a fully connected graph
    for u in g.nodes:
        for v in g.nodes:
            # avoid self-loops
            if u == v:
                continue
            # retrieve the centroids
            cu = np.array(g.nodes[u]['centroid'])
            cv = np.array(g.nodes[v]['centroid'])
            
            # compute the distance
            w = np.linalg.norm(cu-cv, ord=2)
            
            # add the edge
            g.add_edge(u, v, weight=w)

    draw_pos = {}
    for n in g.nodes:
        draw_pos[n] = g.nodes[n]['centroid'][:2]


    # find the minimum weight Hamiltonian path from 'robot' to 'depot'
    permutations = list(itertools.permutations(range(len(g.nodes)-2)))
    best_wt = None
    best_path = None
    for perm in permutations:
        path = ['robot']
        for n in perm:
            path.append(n)
        path.append('depot')
        
        # compute the total weight
        wt = 0
        for i in range(len(path)-1):
            wt = wt + g.edges[path[i], path[i+1]]['weight']
        
        # save the best path
        if best_wt==None or wt < best_wt:
            best_wt = wt
            best_path = path

    print("Best path: {:} ({:})".format(best_path, best_wt))
        
    # ------------------- FOR VISUALIZATION ONLY -------------------

    g_small = nx.Graph()
    draw_pos_small = {}
    for i, pose in enumerate(des_points):
        g_small.add_node(i, pos=pose)
        draw_pos_small[i] = pose[:2]
        
    g_small.add_node('robot')
    g_small.nodes['robot']['pos'] = [0.0, 0.0, 0.0]
    draw_pos_small['robot'] = [0.0, 0.0]

    g_small.add_node('depot')
    g_small.nodes['depot']['pos'] = [0.82, 5.43, 0.0]
    draw_pos_small['depot'] = [0.82, 5.43]

    for u in g_small.nodes:
        for v in g_small.nodes:
            if u == v:
                continue
            pos_u = np.array(g_small.nodes[u]['pos'])
            pos_v = np.array(g_small.nodes[v]['pos'])
            
            w = np.mean(pos_u-pos_v, axis=0)
            
            g_small.add_edge(u, v, weight=w)

    ax = plt.subplot(111)
    nx.draw_networkx_nodes(g_small, pos=draw_pos_small, ax=ax)
    nx.draw_networkx_labels(g_small, pos=draw_pos_small, ax=ax)
    nx.draw_networkx_edges(g_small, pos=draw_pos_small, ax=ax, edge_color="gray", alpha=0.2)

    nx.draw_networkx_nodes(g, pos=draw_pos, nodelist=list(range(len(g.nodes)-2)),
                        node_size=1000, node_color="orange", alpha=0.7)
    nx.draw_networkx_edges(g, pos=draw_pos, ax=ax, edge_color="black", width=2)

    edge_path = []
    for i in range(len(best_path)-1):
        edge_path.append((best_path[i], best_path[i+1]))
    nx.draw_networkx_edges(g, pos=draw_pos, edgelist=edge_path, edge_color="red", width=2)

    plt.savefig("/home/humans/base_pose_opt_multi_dynamic_env_SIM/figures/graph_path_cluster.png")
    
    # ------------------- ------------------- -------------------
    
    # convert the node sequence given in the best_path into weights
    # for the optimization problem
    pose_sequence = best_path[1:len(best_path)-1]
    weights = np.zeros(len(des_position_multi))
    w_acc = 0
    for i in range(len(pose_sequence)-1, -1, -1):
        w = w_acc + 1
        
        for p in g.nodes[pose_sequence[i]]['poses']:
            index = des_points.index(p)
            weights[index] = w
            w_acc = w_acc + w
    
    for i in range(len(des_points)):
        rospy.loginfo("{:} weight: {:}".format(des_points[i], weights[i]))
        
    return weights
        
def compute_weights_greedy(des_position_multi):
    # create a graph
    g = nx.Graph()
    rospy.loginfo("Graph created")
    
    draw_pos = {}
    # add a node associated to the current position of the robot
    g.add_node('robot', coord=[0.0, 0.0])
    rospy.loginfo("Added node 'robot'. Current nodes:")
    rospy.loginfo("{:}: {:}".format(g.nodes, g.nodes['robot']['coord']))
    draw_pos['robot'] = [0.0, 0.0]
    
    # add a node associated to the depot position
    g.add_node('depot', coord=[6.0, -0.32])
    # g.add_node('depot', coord=[0.41, 3.34])
    rospy.loginfo("Added node 'robot'. Current nodes:")
    rospy.loginfo("{:}: {:}".format(g.nodes, g.nodes['robot']['coord']))
    draw_pos['depot'] = [6.0, -0.32]
    # draw_pos['depot'] = [0.41, 3.34]
    
    # add all des_poses as nodes.
    # vetrtex i is associated with i-th pose
    rospy.loginfo("Added nodes associated to the grasp poses. Current nodes:")
    for v, pose in enumerate(des_position_multi):
        # add a node and the pose coordintes
        g.add_node(v, coord=[pose.x, pose.y])
        rospy.loginfo("{:}: {:}".format(v, g.nodes[v]['coord']))
        
        draw_pos[v] = coord=[pose.x, pose.y]
    
    # add edges building a fully connected graph
    rospy.loginfo("Added all edges. Created a fully connected graph")
    for vi in g.nodes:
        for vj in g.nodes:
            # check if they are the same to avoid self loops
            if vi==vj:
                continue
            
            # compute the dist between nodes
            dist = np.linalg.norm(np.array(g.nodes[vi]['coord']) - np.array(g.nodes[vj]['coord']), ord=2)
            
            # add edge with weight (dist)
            g.add_edge(vi, vj, weight=dist, color="black")
    
            # rospy.loginfo("({:},{:}): {:}".format(vi, vj, g.edges[vi, vj]['weight']))
    
    ax1 = plt.subplot(121)
    nx.draw(g, pos=draw_pos, ax=ax1, with_labels=True)
                
    # find the shortest path that traverse all nodes from 'robot' to 'depot'
    path = find_hamiltonian_path_greedy(g)
    rospy.loginfo("Hamiltonian path found from 'robot' to 'depot': {:}".format(path))
    
    edge_path = []
    for i in range(len(path)-1):
        g.edges[path[i], path[i+1]]['color'] = "red"
        edge_path.append((path[i], path[i+1]))
            
    edge_colors = [g.edges[u, v]['color'] for u, v in edge_path]
    
    ax2 = plt.subplot(122)
    nx.draw_networkx_nodes(g, ax=ax2, pos=draw_pos)
    nx.draw_networkx_labels(g, ax=ax2, pos=draw_pos)
    nx.draw_networkx_edges(g, ax=ax2, pos=draw_pos, 
                           edge_color=edge_colors, edgelist=edge_path, width=2)
    plt.savefig("/home/humans/base_pose_opt_multi_dynamic_env_SIM/figures/graph_path_greedy.png")
    
    # convert the node sequence given in the path into weights
    # for the optimization problem
    pose_sequence = path[1:len(path)-1]
    weights = np.zeros(len(des_position_multi))
    for i, node in enumerate(pose_sequence):
        weights[node] = pow(2, len(pose_sequence)-1-i)

    for i in range(len(des_position_multi)):
        rospy.loginfo("pose {:} weight: {:}".format(i, weights[i]))

    return weights
     
def compute_free_2d_space():
    # retrieve the occupancy grid map
    msg = rospy.wait_for_message('/locobot/octomap_server/projected_map', OccupancyGrid)

    # Convert OccupancyGrid to OpenCV image
    width = msg.info.width
    height = msg.info.height
    
    # Convert data to numpy array
    grid_data = np.array(msg.data, dtype=np.uint8).reshape((height, width))

    # values in grid_data are probabilities from 0 to 100, with -1 for unknown
    # - cells with value 0 are free
    # - cells with value 100 are occupied
    # - cells with value -1 are unknown

    # convert unknown cells to occupied
    image = np.where(grid_data == -1, 100, grid_data)
    
    # convert to a grayscale image where black (0) = occupied, white (255) = free
    image = np.uint8(255 - image * 2.55)
    
    # Flip the image vertically to correct ROS to OpenCV coordinate system
    # ROS: origin at bottom-left, Y-axis up
    # OpenCV: origin at top-left, Y-axis down
    image = cv2.flip(image, 0)
    
    # cv2.imshow('Original Occupancy Grid', image)
    # cv2.waitKey(1)
    
    # cv2.imwrite("/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/base_optimization/img/occupancy_grid.png",
    #             image)
    

    # Inflate obstacles with radius d (in meters)
    robot_rad = 0.175
    obstacle_dist = 0.05
    d_m = robot_rad + obstacle_dist
    resolution = msg.info.resolution
    
    # Inflation radius in pixels
    d = int(d_m / resolution)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d+1, 2*d+1))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    # Create binary mask where obstacles are white (occupied cells have low values)
    obstacle_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow('Obstacle Mask', obstacle_mask)
    # cv2.waitKey(1)

    # Dilate obstacles
    inflated_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
    # Find the inflation layer (difference between inflated and original obstacles)
    inflation_layer = cv2.subtract(inflated_mask, obstacle_mask)

    # cv2.imwrite("/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/base_optimization/img/inflated_obstacle.png",
    #             inflation_layer)
  
    # Convert grayscale image to BGR for color visualization
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Color the inflation layer red (BGR format: red = [0, 0, 255])
    image_color[inflation_layer > 0] = [0, 0, 255]
    # Keep original obstacles black
    image_color[obstacle_mask > 0] = [0, 0, 0]
    
    # Display the image
    # cv2.imshow('Inflation Layer', image_color)
    # cv2.waitKey(1)

    # create an image with all obstacels inflated
    # (only free space is visible as white)
    image_inflated = image
    image_inflated[inflated_mask > 0] = 0

    # cv2.imshow('Inflated Occupancy Grid', image_inflated)
    # cv2.waitKey(1000)
    
    # cv2.imwrite("/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/base_optimization/img/inflated_grid.png",
    #             image_inflated)

    # Keep only the connected white space containing the robot's position
    # Retrieve robot position in map frame
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    tf_buffer.can_transform("octomap_frame", "locobot/base_footprint", rospy.Time(0))
    transform = tf_buffer.lookup_transform("octomap_frame", "locobot/base_footprint", rospy.Time(0), rospy.Duration(1))
    robot_x = transform.transform.translation.x
    robot_y = transform.transform.translation.y
    
    # Convert robot position from map frame to pixel coordinates
    origin_x = msg.info.origin.position.x
    origin_y = msg.info.origin.position.y
    robot_pixel_x = int((robot_x - origin_x) / resolution)
    robot_pixel_y = int((robot_y - origin_y) / resolution)
    
    # Flip y coordinate because we flipped the image
    robot_pixel_y = height - 1 - robot_pixel_y
        
    # Flood fill from robot position
    free_space = image_inflated.copy()
    cv2.floodFill(free_space, None, (robot_pixel_x, robot_pixel_y), 128)
        
    # Keep only the flooded region (value 128), set everything else to black (0)
    image_inflated_robot_space = np.where(free_space == 128, 255, 0).astype(np.uint8)
    cv2.circle(image_inflated_robot_space, (robot_pixel_x, robot_pixel_y), 3, (190), -1)
    
    # cv2.imshow('Robot Free Space', image_inflated_robot_space)
    # cv2.waitKey(0)
    # cv2.startWindowThread()

    # cv2.imwrite("/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/base_optimization/img/robot_free_space.png",
    #             image_inflated_robot_space)

    return msg.info, image_inflated_robot_space

def handle_des_EE_pose_multi(pose_array_msg):
    rospy.logwarn("Received")
    # retrieve the set of desired points
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
    
    # compute the weights for each pose
    # weights = compute_weights_greedy(des_position_multi)
    # weights = compute_weights_clustering(des_position_multi)
    weights = np.ones(len(des_position_multi))
    
    # compute the free 2D space where the mobile base can move
    occupancy_grid_info, free_space_2d = compute_free_2d_space()

    # request the conversion from octomap to point cloud
    rospy.loginfo(
        "Waiting for service '/locobot/octomap2cloud_converter_srv' ...")
    rospy.wait_for_service("/locobot/octomap2cloud_converter_srv")
    octomap2cloud_srv = rospy.ServiceProxy(
        "/locobot/octomap2cloud_converter_srv", octomap2cloud)

    rospy.loginfo(
        "Sending a service request to '/locobot/octomap2cloud_converter_srv'...")
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
