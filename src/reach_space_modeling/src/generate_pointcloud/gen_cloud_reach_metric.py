# import tkinter as tk
# import numpy as np
# from scipy.spatial.transform import Rotation
# from generate_pointcloud.gen_cloud_GUI import GenereatePointCloud
# 
# import torch
# import time

# import tkinter as tk
# import numpy as np
# import time
# from scipy.spatial.transform import Rotation
# from sensor_msgs.msg import PointCloud2, PointField
# from visualization_msgs.msg import Marker, MarkerArray
# from std_msgs.msg import Header
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from mpl_toolkits.mplot3d import Axes3D
# import rospy
# from tqdm import tqdm

# import PyKDL
# from urdf_parser_py.urdf import URDF
# from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
# import pybullet


# class GenereatePointCloudWithMetric(GenereatePointCloud):
#     def __init__(self) -> None:
#         super().__init__()

#     def create_ros_node(self):
#         rospy.init_node('reachability_pointcloud_publisher', anonymous=True)
#         self.pub_ellipsoid_inn = rospy.Publisher(
#             '/viz_reachability_ellipsoid_inn', Marker, queue_size=10)
#         self.pub_ellipsoid_out = rospy.Publisher(
#             '/viz_reachability_ellipsoid_out', Marker, queue_size=10)
#         self.pub_center = rospy.Publisher(
#             '/viz_ellipsoid_center', Marker, queue_size=10)
#         self.pub_cloud = rospy.Publisher(
#             '/viz_pointcloud', MarkerArray, queue_size=10)

#     def generate_reachability_index(self):
#         # define the parameter of the sphere
#         c = np.zeros(3)
#         r = 0.05

#         # sample the sphere centered in the origin.
#         # since the sampling algorithm has a deterministic behaviour,
#         # the coordinate of the samples with respect to the reference system
#         # fixed in the center of the sphere will be always the same
#         sphere_samples = self.sample_sphere_fibonacci_grid(
#             center=c, radius=r, n_samples=20)
#         sphere_samples = np.array(sphere_samples)
#         rospy.loginfo("Generated sampling of each sphere...")

#         # generate a pose for each samples.
#         # the orientation of the poses is always the same
#         pose_angles_RPY = self.gen_poses(
#             center=c, radius=r, sphere_samples=sphere_samples)
#         pose_angles_RPY = np.array(pose_angles_RPY)
#         rospy.loginfo("Generated all target poses...")

#         # Load the robot chain from a URDF file for the KDL library
#         self.init_KDL_model()

#         # init the KDL solvers
#         self.init_KDL_solvers()

#         # start pybullet for self-collision checking
#         # pybullet.connect(pybullet.GUI, options="--width=2000 --height=1000")
#         pybullet.connect(pybullet.DIRECT)

#         # init a pybullet model of the robot
#         self.pybullet_robot = pybullet.loadURDF(
#             self.urdf_file_path, useFixedBase=True)
#         # input("Press Enter to conitnue...")

#         # create a list of lists for links that collides by default
#         self.default_self_collision()

#         # define a (dummy) initial guess for the joints
#         initial_jnts_val = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())
#         for i in range(self.kdl_chain.getNrOfJoints()):
#             initial_jnts_val[i] = 0.0

#         # store the solution to the IK problem
#         calc_jnt_pos = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())

#         # create an array for the score of each point
#         self.points_reach_measure = np.zeros(self.points.shape[0])

#         # rospy.logwarn(self.kdl_chain.getNrOfJoints())
#         # rospy.logwarn(pybullet.getNumJoints(self.pybullet_robot))
        
#         # rospy.logwarn(pybullet.getNumJoints(self.pybullet_robot))
#         # for i in range(pybullet.getNumJoints(self.pybullet_robot)):
#         #     joint_info = pybullet.getJointInfo(self.pybullet_robot, i)
#         #     joint_name = joint_info[1].decode("utf-8")
#         #     rospy.logwarn(f"Joint index: {i}, Joint name: {joint_name}")

#         # iterate over all points and poses
#         for i, pnt in tqdm(enumerate(self.points),
#                            total=self.points.shape[0],
#                            desc="Point examinated",
#                            ncols=100):
#             for pose in pose_angles_RPY:
#                 # create a target pose
#                 target_pose = PyKDL.Frame(PyKDL.Rotation.RPY(pose[0], pose[1], pose[2]),
#                                           PyKDL.Vector(pnt[0], pnt[1], pnt[2]))

#                 # find a solution fo the IK problem
#                 result = self.kdl_ik_solver.CartToJnt(initial_jnts_val,
#                                                       target_pose,
#                                                       calc_jnt_pos)

#                 # if a solution is found, increment the score of the point
#                 if result >= 0:
#                     # self.points_reach_measure[i] = self.points_reach_measure[i] + 1

#                     for index in range(11):
#                         pybullet.resetJointState(
#                             self.pybullet_robot, index, targetValue=0.001)

#                     # for the used joints, set the found value
#                     for index in range(self.kdl_chain.getNrOfJoints()):
#                         pybullet.resetJointState(
#                             self.pybullet_robot, index+11, targetValue=calc_jnt_pos[index])

#                     # se to 0.0 the others
#                     for index in range(self.kdl_chain.getNrOfJoints()+11,
#                                    pybullet.getNumJoints(self.pybullet_robot)):
#                         pybullet.resetJointState(
#                             self.pybullet_robot, index, targetValue=0.001)

#                     # check for self-collisions
#                     contacts_flag = False
#                     for jnt_i in range(pybullet.getNumJoints(self.pybullet_robot)):
#                         for jnt_j in range(pybullet.getNumJoints(self.pybullet_robot)):
#                             # if the links collide by default continue
#                             if jnt_j in self.default_coll[jnt_i]:
#                                 continue
#                             contacts = pybullet.getClosestPoints(bodyA=self.pybullet_robot,
#                                                                        bodyB=self.pybullet_robot,
#                                                                        distance=0.001,
#                                                                        linkIndexA=jnt_i,
#                                                                        linkIndexB=jnt_j)
#                             # print(contacts)
#                             if contacts:
#                                 # print(f"Collisione rilevata tra il link {jnt_i} e il link {jnt_j}!")
#                                 # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 0, 0, 1])
#                                 # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 0, 0, 1])
#                                 # input()
#                                 # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 1, 1, 1])
#                                 # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 1, 1, 1])
#                                 contacts_flag = True
#                                 break
#                         if contacts_flag:
#                             break

#                     # the reachability measure for point i is incremented only if:
#                     #       1- a solution to the IK problem exist
#                     #       2- the joint configuration found does not lead to any self-collision
#                     #          (besides those that occure due to the way links are attached)
#                     if not contacts_flag:
#                         self.points_reach_measure[i] = self.points_reach_measure[i] + 1
#                         # rospy.logwarn("Nessuna collisione per questa posa YEEEEEEEEE!!!!")

#         pybullet.disconnect()

#     def init_KDL_model(self):
#         # read the urdf file
#         with open(self.urdf_file_path, "r") as f:
#             urdf_string = f.read()

#         # load the robot model
#         self.kdl_robot = URDF.from_xml_string(urdf_string)

#         # create a KDL tree
#         kdl_tree = kdl_tree_from_urdf_model(self.kdl_robot)

#         # select the base_link and the end_link of the desired kinematic chain
#         arm_frt_joint = self.robot.joint_map.get(self.arm_frt_j_name)
#         base_link = arm_frt_joint.parent
#         rospy.loginfo(
#             "Starting link for the PyKDL chian: {}".format(base_link))

#         wrist_lst_joint = self.robot.joint_map.get(self.wrist_lst_j_name)
#         end_link = wrist_lst_joint.child
#         rospy.loginfo("End link for the PyKDL chian: {}".format(end_link))

#         self.kdl_chain = kdl_tree.getChain(base_link, end_link)

#     def init_KDL_solvers(self):
#         # set the joint limits
#         joint_limits_lower = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())
#         joint_limits_upper = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())

#         # set the joint limits
#         for i in range(self.kdl_chain.getNrOfSegments()):
#             segment = self.kdl_chain.getSegment(i)

#             joint = segment.getJoint()

#             # check if it is not a fixed joint
#             if joint.getType() != 8:
#                 joint_limits_lower[i] = self.kdl_robot.joint_map.get(
#                     joint.getName()).limit.lower
#                 joint_limits_upper[i] = self.kdl_robot.joint_map.get(
#                     joint.getName()).limit.upper
#             else:
#                 joint_limits_lower[i] = 0.0
#                 joint_limits_upper[i] = 0.0

#         # init the FK solver
#         self.kdl_fk_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)

#         # init the Ik solver
#         self.kdl_ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)
#         self.kdl_ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kdl_chain)

#     def default_self_collision(self):
#         # create a list of lists
#         self.default_coll = []
#         for jnt_i in range(pybullet.getNumJoints(self.pybullet_robot)):
#             self.default_coll.append([])

#         # iterate over all joints
#         for jnt_i in range(pybullet.getNumJoints(self.pybullet_robot)):
#             # for jnt_j in range(jnt_i+1, pybullet.getNumJoints(self.pybullet_robot)):
#             for jnt_j in range(pybullet.getNumJoints(self.pybullet_robot)):
#                 contacts = pybullet.getClosestPoints(bodyA=self.pybullet_robot,
#                                                      bodyB=self.pybullet_robot,
#                                                      distance=0.002,
#                                                      linkIndexA=jnt_i,
#                                                      linkIndexB=jnt_j)
#                 if contacts:
#                     self.default_coll[jnt_i].append(jnt_j)
#                     self.default_coll[jnt_j].append(jnt_i)
#                     # print(f"Collisione rilevata tra il link {jnt_i} e il link {jnt_j}!")
#                     # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 0, 0, 1])
#                     # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 0, 0, 1])
#                     # input()
#                     # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 1, 1, 1])
#                     # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 1, 1, 1])

#         # for i in range(len(self.default_coll)):
#         #     print("Link {:d} collids with: ".format(i), end="")
#         #     print(self.default_coll[i])

#     def vis_cloud_with_measure(self):
#         # figure of the reach cloud with the manipulability measure
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')

#         colors = ['red', 'yellow', 'lightgreen', 'blue']  # low to high
#         custom_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

#         color_values = (self.points_reach_measure - np.min(self.points_reach_measure)) / \
#             (np.max(self.points_reach_measure) - np.min(self.points_reach_measure))
#         sc = ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
#                         c=self.points_reach_measure, cmap=custom_cmap, s=10)

#         cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
#         cmap = sc.get_cmap()
#         norm = sc.norm
#         colors = cmap(norm(self.points_reach_measure))

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.axis("equal")

#         # figure of a section (xz plane) of the reachability cloud with measure
#         tolerance = 0.01
#         mask = (self.points[:, 1] >= 0 -
#                 tolerance) & (self.points[:, 1] <= 0+tolerance)

#         sec_points = self.points[mask, :]
#         # sec_reach_measure = self.points_reach_measure.tolist()
#         sec_reach_measure = [self.points_reach_measure[i]
#                              for i in range(len(mask)) if mask[i]]
#         fig2 = plt.figure(figsize=(8, 6))
#         sc = plt.scatter(sec_points[:, 0], sec_points[:, 2],
#                          c=sec_reach_measure, cmap='plasma_r', s=10)

#         cbar = plt.colorbar(sc)
#         plt.xlabel("X")
#         plt.ylabel("Z")

#         plt.show()

#         return colors

#     def vis_cloud_and_poses(self):
#         c = np.zeros(3)
#         r = 0.05
#         sphere_samples = self.sample_sphere_fibonacci_grid(
#             center=c, radius=r, n_samples=20)
#         sphere_samples = np.array(sphere_samples)
#         rospy.loginfo("Generated sampling of each sphere...")

#         # generate a pose for each samples.
#         # the orientation of the poses is always the same
#         pose_angles_RPY = gen_cloud.gen_poses(
#             center=c, radius=r, sphere_samples=sphere_samples)
#         pose_angles_RPY = np.array(pose_angles_RPY)
#         rospy.loginfo("Generated all target poses...")

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')

#         # for each reachable point...
#         for i, point in enumerate(self.points):            
#             # parameter of the sphere
#             c = point
#             r = 0.05

#             # plot the center
#             ax.scatter(c[0],
#                         c[1],
#                         c[2],
#                         c='blue',
#                         alpha=1,
#                         s=20)

#             # plot an arrow for each pose
#             ax.quiver(sphere_samples[:, 0] + c[0],
#                       sphere_samples[:, 1] + c[1],
#                       sphere_samples[:, 2] + c[2],
#                       -sphere_samples[:, 0],
#                       -sphere_samples[:, 1],
#                       -sphere_samples[:, 2],
#                       length=0.5,
#                       arrow_length_ratio=1)

#             ax.scatter(sphere_samples[:, 0] + c[0],
#                        sphere_samples[:, 1] + c[1],
#                        sphere_samples[:, 2] + c[2],
#                        c='red',
#                        s=5)
            
#             u = np.linspace(0, 2 * np.pi, 50)
#             v = np.linspace(0, np.pi, 50)
#             x = c[0] + r * np.outer(np.cos(u), np.sin(v))
#             y = c[1] + r * np.outer(np.sin(u), np.sin(v))
#             z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))

#             # Plot the surface
#             ax.plot_surface(x, y, z, color='lightgray', alpha=0.1, edgecolor='none')
#             ax.set_aspect("equal")
#             ax.set_axis_off()

#         plt.show()

#     def sample_sphere_azimut_elevation(self, center, radius):
#         # define the center
#         x, y, z = center[0], center[1], center[2]

#         # define a linspace for the azimuth
#         azimuth = np.linspace(0, 2*np.pi, 20)

#         # define a linspace for the elevation
#         elevation = np.linspace(0, np.pi, 10)

#         # create the grid of points
#         samples = []
#         for a in azimuth:
#             for e in elevation:
#                 xs = radius * np.cos(a) * np.sin(e) + x
#                 ys = radius * np.sin(a) * np.sin(e) + y
#                 zs = radius * np.cos(e) + z
#                 s = [xs, ys, zs]
#                 samples.append(s)

#         return samples

#     def sample_sphere_archimede(self, center, radius, n_samples=50):
#         x, y, z = center[0], center[1], center[2]

#         samples = []

#         for i in range(int(n_samples/2)):
#             theta = 2 * np.pi * np.random.uniform(-1, 1)
#             zs = radius * np.random.uniform(0, 1) + z
#             xs = radius * np.sqrt(1 - zs**2) * np.cos(theta) + x
#             ys = radius * np.sqrt(1 - zs**2) * np.sin(theta) + y

#             s1 = [xs, ys, zs]
#             s2 = [-xs, -ys, -zs]
#             samples.append(s1)
#             samples.append(s2)

#         return samples

#     def sample_sphere_fibonacci_grid(self, center, radius, n_samples=50):
#         x, y, z = center[0], center[1], center[2]

#         samples = []

#         r_phi = np.pi * (np.sqrt(5.) - 1.)

#         for i in range(n_samples):
#             ys = 1 - (i/float(n_samples - 1)) * 2

#             r = np.sqrt(1 - ys**2)
#             theta = r_phi * i

#             xs = np.cos(theta) * r
#             zs = np.sin(theta) * r

#             s = [x + radius * xs, y + radius * ys, z + radius * zs]
#             samples.append(s)

#         return samples

#     def sample_sphere_golden_spiral(self, center, radius, n_samples=50):
#         x, y, z = center[0], center[1], center[2]

#         indices = np.arange(0, n_samples, dtype=float) + 0.5

#         phi = np.arccos(1 - 2*indices/n_samples)
#         theta = np.pi * (1 + 5**0.5) * indices

#         Xs = np.cos(theta) * np.sin(phi)
#         Ys = np.sin(theta) * np.sin(phi)
#         Zs = np.cos(phi)

#         samples = [[x + radius*Xs[i], y + radius*Ys[i], z + radius*Zs[i]]
#                    for i in range(n_samples)]

#         return samples

#     def vis_sample_sphere(self, center, radius, sphere_samples):
#         x, y, z = center[0], center[1], center[2]

#         alpha = np.linspace(0, 2*np.pi, 50)
#         theta = np.linspace(0, np.pi, 50)

#         A, T = np.meshgrid(alpha, theta)

#         X = radius * np.cos(A) * np.sin(T) + x
#         Y = radius * np.sin(A) * np.sin(T) + y
#         Z = radius * np.cos(T) + z

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection="3d")
#         ax.plot_surface(X, Y, Z, alpha=0.2)

#         sphere_samples = np.array(sphere_samples)
#         ax.scatter(sphere_samples[:, 0], sphere_samples[:,
#                    1], sphere_samples[:, 2], c='red')

#         ax.set_aspect("equal")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.set_title("Visualization of a sphere and its surface samples")

#         plt.show()

#     def gen_poses(self, center, radius, sphere_samples):
#         # center coordinates
#         xc, yc, zc = center[0], center[1], center[2]
#         '''
#             RPY sono gli angoli rispetto al sistema di riferimento in cui sono
#             stati definiti i centri delle sfere (punti raggiungibili)
#         '''
#         pose_angles_RPY = []

#         for i in range(len(sphere_samples)):
#             xs, ys, zs = sphere_samples[i,0], sphere_samples[i, 1], sphere_samples[i, 2]

#             # compute the vector parallel to the line going from a sample
#             # towards the center of the sphere
#             v = [xc-xs, yc-ys, zc-zs]

#             # normalize the vector
#             v_norm = v / np.linalg.norm(v, ord=2)

#             # compute the yaw angle
#             yaw = np.arctan2(v_norm[1], v_norm[0])

#             # compute the pitch angle
#             pitch = -np.arctan2(v_norm[2],
#                                 np.sqrt(v_norm[1]**2 + v_norm[0]**2))

#             # set the roll angle to 0
#             roll = 0.

#             pose_angles_RPY.append([roll, pitch, yaw])

#         return pose_angles_RPY

#     def vis_sphere_sample_poses(self, center, radius, sphere_samples, pose_angles_RPY):
#         x, y, z = center[0], center[1], center[2]

#         alpha = np.linspace(0, 2*np.pi, 50)
#         theta = np.linspace(0, np.pi, 50)

#         A, T = np.meshgrid(alpha, theta)

#         X = radius * np.cos(A) * np.sin(T) + x
#         Y = radius * np.sin(A) * np.sin(T) + y
#         Z = radius * np.cos(T) + z

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection="3d")

#         # plot the sphere
#         ax.plot_surface(X, Y, Z, alpha=0.2)

#         # plot the surface samples
#         sphere_samples = np.array(sphere_samples)
#         ax.scatter(sphere_samples[:, 0], sphere_samples[:,
#                    1], sphere_samples[:, 2], c='red')

#         # plot the generated pose
#         for i in range(len(pose_angles_RPY)):
#             rot_mat = Rotation.from_euler(
#                 'xyz', pose_angles_RPY[i, :], degrees=False).as_matrix()
#             dir = np.array([radius, 0, 0])
#             end = np.dot(rot_mat, dir.T)

#             ax.quiver(sphere_samples[i, 0], sphere_samples[i, 1], sphere_samples[i, 2],
#                       end[0], end[1], end[2])

#         ax.set_aspect("equal")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.set_title(
#             "Visualization of a sphere, its surface samples and the generated poses")

#         plt.show()

#     def create_pointcloud_msg(self, colors):        
#         # create the MarkerArray message
#         markArray_msg = MarkerArray()
        
#         for i in range(self.points.shape[0]):
#             marker = Marker()
#             marker.id = i
#             marker.header.frame_id = "map"
#             marker.header.stamp = rospy.Time.now()
#             marker.type = Marker.SPHERE
#             marker.pose.position.x = self.points[i,0]
#             marker.pose.position.y = self.points[i,1]
#             marker.pose.position.z = self.points[i,2]
#             marker.pose.orientation.w = 1.0
#             marker.scale.x = 0.05
#             marker.scale.y = 0.05
#             marker.scale.z = 0.05
#             marker.color.r = colors[i,0]
#             marker.color.g = colors[i,1]
#             marker.color.b = colors[i,2]
#             marker.color.a = 0.7
            
#             markArray_msg.markers.append(marker)
        
#         return markArray_msg
            
#     def publish_pointcloud_msg(self, colors):
#         markArray_msg = self.create_pointcloud_msg(colors)
#         self.pub_cloud.publish(markArray_msg)
#         rospy.sleep(0.5)

# if __name__ == "__main__":
#     gen_cloud = GenereatePointCloudWithMetric()
#     gen_cloud.create_ros_node()
#     gen_cloud.create_GUI()

#     # gen_cloud.from_extern = True
#     # gen_cloud.urdf_file_path = "/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/reach_space_modeling/src/generate_pointcloud/model/mobile_wx250s.urdf"
#     # gen_cloud.parse_urdf()
#     # gen_cloud.wrist_lst_j_name = "wrist_rotate"
#     # gen_cloud.arm_lst_j_name = "elbow"
#     # gen_cloud.arm_frt_j_name = "waist"
#     # gen_cloud.num_samples = 10
#     # gen_cloud.generate_point_cloud()
#     rospy.loginfo("Reachability point cloud created...")

#     gen_cloud.generate_reachability_index()
#     colors = gen_cloud.vis_cloud_with_measure()
#     gen_cloud.publish_pointcloud_msg(colors)
#     # gen_cloud.vis_cloud_and_poses()

import tkinter as tk
import numpy as np
from scipy.spatial.transform import Rotation
from generate_pointcloud.gen_cloud_GUI import GenereatePointCloud

import torch
import time

import tkinter as tk
import numpy as np
import time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import rospy
from tqdm import tqdm

import PyKDL
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import pybullet


class GenereatePointCloudWithMetric(GenereatePointCloud):
    def __init__(self) -> None:
        super().__init__()

    def create_ros_node(self):
        rospy.init_node('reachability_pointcloud_publisher', anonymous=True)
        self.pub_ellipsoid_inn = rospy.Publisher(
            '/viz_reachability_ellipsoid_inn', Marker, queue_size=10)
        self.pub_ellipsoid_out = rospy.Publisher(
            '/viz_reachability_ellipsoid_out', Marker, queue_size=10)
        self.pub_center = rospy.Publisher(
            '/viz_ellipsoid_center', Marker, queue_size=10)
        self.pub_cloud = rospy.Publisher(
            '/viz_pointcloud', MarkerArray, queue_size=10)

    def generate_reachability_index(self):
        # define the parameter of the sphere
        c = np.zeros(3)
        r = 0.05

        # sample the sphere centered in the origin.
        # since the sampling algorithm has a deterministic behaviour,
        # the coordinate of the samples with respect to the reference system
        # fixed in the center of the sphere will be always the same
        sphere_samples = self.sample_sphere_fibonacci_grid(
            center=c, radius=r, n_samples=20)
        sphere_samples = np.array(sphere_samples)
        rospy.loginfo("Generated sampling of each sphere...")

        # generate a pose for each samples.
        # the orientation of the poses is always the same
        pose_angles_RPY = self.gen_poses(
            center=c, radius=r, sphere_samples=sphere_samples)
        pose_angles_RPY = np.array(pose_angles_RPY)
        rospy.loginfo("Generated all target poses...")

        # Load the robot chain from a URDF file for the KDL library
        self.init_KDL_model()

        # init the KDL solvers
        self.init_KDL_solvers()

        # start pybullet for self-collision checking
        # pybullet.connect(pybullet.GUI, options="--width=2000 --height=1000")
        pybullet.connect(pybullet.DIRECT)

        # init a pybullet model of the robot
        self.pybullet_robot = pybullet.loadURDF(
            self.urdf_file_path, useFixedBase=True)
        # input("Press Enter to conitnue...")

        # create a list of lists for links that collides by default
        self.default_self_collision()

        # define a (dummy) initial guess for the joints
        initial_jnts_val = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())
        for i in range(self.kdl_chain.getNrOfJoints()):
            initial_jnts_val[i] = 0.0

        # store the solution to the IK problem
        calc_jnt_pos = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())

        # create an array for the score of each point
        self.points_reach_measure = np.zeros(self.points.shape[0])

        # iterate over all points and poses
        for i, pnt in tqdm(enumerate(self.points),
                           total=self.points.shape[0],
                           desc="Point examinated",
                           ncols=100):
            for pose in pose_angles_RPY:
                # create a target pose
                target_pose = PyKDL.Frame(PyKDL.Rotation.RPY(pose[0], pose[1], pose[2]),
                                          PyKDL.Vector(pnt[0], pnt[1], pnt[2]))

                # find a solution fo the IK problem
                result = self.kdl_ik_solver.CartToJnt(initial_jnts_val,
                                                      target_pose,
                                                      calc_jnt_pos)

                # if a solution is found, increment the score of the point
                if result >= 0:
                    # self.points_reach_measure[i] = self.points_reach_measure[i] + 1

                    # for the used joints, set the found value
                    for index in range(self.kdl_chain.getNrOfJoints()):
                        pybullet.resetJointState(
                            self.pybullet_robot, index, targetValue=calc_jnt_pos[index])

                    # se to 0.0 the others
                    for index in range(self.kdl_chain.getNrOfJoints()+1,
                                   pybullet.getNumJoints(self.pybullet_robot)):
                        pybullet.resetJointState(
                            self.pybullet_robot, index, targetValue=0.001)

                    # check for self-collisions
                    contacts_flag = False
                    for jnt_i in range(pybullet.getNumJoints(self.pybullet_robot)):
                        for jnt_j in range(jnt_i+1, pybullet.getNumJoints(self.pybullet_robot)):
                            # if the links collide by default continue
                            if jnt_j in self.default_coll[jnt_i]:
                                continue
                            contacts = pybullet.getClosestPoints(bodyA=self.pybullet_robot,
                                                                       bodyB=self.pybullet_robot,
                                                                       distance=0.001,
                                                                       linkIndexA=jnt_i,
                                                                       linkIndexB=jnt_j)
                            # print(contacts)
                            if contacts:
                                # print(f"Collisione rilevata tra il link {jnt_i} e il link {jnt_j}!")
                                # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 0, 0, 1])
                                # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 0, 0, 1])
                                # input()
                                # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 1, 1, 1])
                                # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 1, 1, 1])
                                contacts_flag = True
                                break
                        if contacts_flag:
                            break

                    # the reachability measure for point i is incremented only if:
                    #       1- a solution to the IK problem exist
                    #       2- the joint configuration found does not lead to any self-collision
                    #          (besides those that occure due to the way links are attached)
                    if not contacts_flag:
                        self.points_reach_measure[i] = self.points_reach_measure[i] + 1

        pybullet.disconnect()

    def init_KDL_model(self):
        # read the urdf file
        with open(self.urdf_file_path, "r") as f:
            urdf_string = f.read()

        # load the robot model
        self.kdl_robot = URDF.from_xml_string(urdf_string)

        # create a KDL tree
        kdl_tree = kdl_tree_from_urdf_model(self.kdl_robot)

        # select the base_link and the end_link of the desired kinematic chain
        arm_frt_joint = self.robot.joint_map.get(self.arm_frt_j_name)
        base_link = arm_frt_joint.parent
        rospy.loginfo(
            "Starting link for the PyKDL chian: {}".format(base_link))

        wrist_lst_joint = self.robot.joint_map.get(self.wrist_lst_j_name)
        end_link = wrist_lst_joint.child
        rospy.loginfo("End link for the PyKDL chian: {}".format(end_link))

        self.kdl_chain = kdl_tree.getChain(base_link, end_link)

    def init_KDL_solvers(self):
        # set the joint limits
        joint_limits_lower = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())
        joint_limits_upper = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())

        # set the joint limits
        for i in range(self.kdl_chain.getNrOfSegments()):
            segment = self.kdl_chain.getSegment(i)

            joint = segment.getJoint()

            # check if it is not a fixed joint
            if joint.getType() != 8:
                joint_limits_lower[i] = self.kdl_robot.joint_map.get(
                    joint.getName()).limit.lower
                joint_limits_upper[i] = self.kdl_robot.joint_map.get(
                    joint.getName()).limit.upper
            else:
                joint_limits_lower[i] = 0.0
                joint_limits_upper[i] = 0.0

        # init the FK solver
        self.kdl_fk_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)

        # init the Ik solver
        self.kdl_ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)
        self.kdl_ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kdl_chain)

    def default_self_collision(self):
        # create a list of lists
        self.default_coll = []
        for jnt_i in range(pybullet.getNumJoints(self.pybullet_robot)):
            self.default_coll.append([])

        # iterate over all joints
        for jnt_i in range(pybullet.getNumJoints(self.pybullet_robot)):
            for jnt_j in range(jnt_i+1, pybullet.getNumJoints(self.pybullet_robot)):
                contacts = pybullet.getClosestPoints(bodyA=self.pybullet_robot,
                                                     bodyB=self.pybullet_robot,
                                                     distance=0.002,
                                                     linkIndexA=jnt_i,
                                                     linkIndexB=jnt_j)
                if contacts:
                    self.default_coll[jnt_i].append(jnt_j)
                    self.default_coll[jnt_j].append(jnt_i)
                    # print(f"Collisione rilevata tra il link {jnt_i} e il link {jnt_j}!")
                    # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 0, 0, 1])
                    # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 0, 0, 1])
                    # input()
                    # pybullet.changeVisualShape(self.pybullet_robot, jnt_i, rgbaColor=[1, 1, 1, 1])
                    # pybullet.changeVisualShape(self.pybullet_robot, jnt_j, rgbaColor=[1, 1, 1, 1])

        # for i in range(len(self.default_coll)):
        #     print("Link {:d} collids with: ".format(i), end="")
        #     print(self.default_coll[i])

    def vis_cloud_with_measure(self):
        # figure of the reach cloud with the manipulability measure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red', 'yellow', 'lightgreen', 'blue']  # low to high
        custom_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

        color_values = (self.points_reach_measure - np.min(self.points_reach_measure)) / \
            (np.max(self.points_reach_measure) - np.min(self.points_reach_measure))
        sc = ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                        c=self.points_reach_measure, cmap=custom_cmap, s=10)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
        cmap = sc.get_cmap()
        norm = sc.norm
        colors = cmap(norm(self.points_reach_measure))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis("equal")

        # figure of a section (xz plane) of the reachability cloud with measure
        tolerance = 0.01
        mask = (self.points[:, 1] >= 0 -
                tolerance) & (self.points[:, 1] <= 0+tolerance)

        sec_points = self.points[mask, :]
        # sec_reach_measure = self.points_reach_measure.tolist()
        sec_reach_measure = [self.points_reach_measure[i]
                             for i in range(len(mask)) if mask[i]]
        fig2 = plt.figure(figsize=(8, 6))
        sc = plt.scatter(sec_points[:, 0], sec_points[:, 2],
                         c=sec_reach_measure, cmap='plasma_r', s=10)

        cbar = plt.colorbar(sc)
        plt.xlabel("X")
        plt.ylabel("Z")

        # plt.show()

        return colors

    def vis_cloud_and_poses(self):
        c = np.zeros(3)
        r = 0.05
        sphere_samples = self.sample_sphere_fibonacci_grid(
            center=c, radius=r, n_samples=20)
        sphere_samples = np.array(sphere_samples)
        rospy.loginfo("Generated sampling of each sphere...")

        # generate a pose for each samples.
        # the orientation of the poses is always the same
        pose_angles_RPY = gen_cloud.gen_poses(
            center=c, radius=r, sphere_samples=sphere_samples)
        pose_angles_RPY = np.array(pose_angles_RPY)
        rospy.loginfo("Generated all target poses...")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # for each reachable point...
        for i, point in enumerate(self.points):            
            # parameter of the sphere
            c = point
            r = 0.05

            # plot the center
            ax.scatter(c[0],
                        c[1],
                        c[2],
                        c='blue',
                        alpha=1,
                        s=20)

            # plot an arrow for each pose
            ax.quiver(sphere_samples[:, 0] + c[0],
                      sphere_samples[:, 1] + c[1],
                      sphere_samples[:, 2] + c[2],
                      -sphere_samples[:, 0],
                      -sphere_samples[:, 1],
                      -sphere_samples[:, 2],
                      length=0.5,
                      arrow_length_ratio=1)

            ax.scatter(sphere_samples[:, 0] + c[0],
                       sphere_samples[:, 1] + c[1],
                       sphere_samples[:, 2] + c[2],
                       c='red',
                       s=5)
            
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x = c[0] + r * np.outer(np.cos(u), np.sin(v))
            y = c[1] + r * np.outer(np.sin(u), np.sin(v))
            z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))

            # Plot the surface
            ax.plot_surface(x, y, z, color='lightgray', alpha=0.1, edgecolor='none')
            ax.set_aspect("equal")
            ax.set_axis_off()

        # plt.show()

    def sample_sphere_azimut_elevation(self, center, radius):
        # define the center
        x, y, z = center[0], center[1], center[2]

        # define a linspace for the azimuth
        azimuth = np.linspace(0, 2*np.pi, 20)

        # define a linspace for the elevation
        elevation = np.linspace(0, np.pi, 10)

        # create the grid of points
        samples = []
        for a in azimuth:
            for e in elevation:
                xs = radius * np.cos(a) * np.sin(e) + x
                ys = radius * np.sin(a) * np.sin(e) + y
                zs = radius * np.cos(e) + z
                s = [xs, ys, zs]
                samples.append(s)

        return samples

    def sample_sphere_archimede(self, center, radius, n_samples=50):
        x, y, z = center[0], center[1], center[2]

        samples = []

        for i in range(int(n_samples/2)):
            theta = 2 * np.pi * np.random.uniform(-1, 1)
            zs = radius * np.random.uniform(0, 1) + z
            xs = radius * np.sqrt(1 - zs**2) * np.cos(theta) + x
            ys = radius * np.sqrt(1 - zs**2) * np.sin(theta) + y

            s1 = [xs, ys, zs]
            s2 = [-xs, -ys, -zs]
            samples.append(s1)
            samples.append(s2)

        return samples

    def sample_sphere_fibonacci_grid(self, center, radius, n_samples=50):
        x, y, z = center[0], center[1], center[2]

        samples = []

        r_phi = np.pi * (np.sqrt(5.) - 1.)

        for i in range(n_samples):
            ys = 1 - (i/float(n_samples - 1)) * 2

            r = np.sqrt(1 - ys**2)
            theta = r_phi * i

            xs = np.cos(theta) * r
            zs = np.sin(theta) * r

            s = [x + radius * xs, y + radius * ys, z + radius * zs]
            samples.append(s)

        return samples

    def sample_sphere_golden_spiral(self, center, radius, n_samples=50):
        x, y, z = center[0], center[1], center[2]

        indices = np.arange(0, n_samples, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices/n_samples)
        theta = np.pi * (1 + 5**0.5) * indices

        Xs = np.cos(theta) * np.sin(phi)
        Ys = np.sin(theta) * np.sin(phi)
        Zs = np.cos(phi)

        samples = [[x + radius*Xs[i], y + radius*Ys[i], z + radius*Zs[i]]
                   for i in range(n_samples)]

        return samples

    def vis_sample_sphere(self, center, radius, sphere_samples):
        x, y, z = center[0], center[1], center[2]

        alpha = np.linspace(0, 2*np.pi, 50)
        theta = np.linspace(0, np.pi, 50)

        A, T = np.meshgrid(alpha, theta)

        X = radius * np.cos(A) * np.sin(T) + x
        Y = radius * np.sin(A) * np.sin(T) + y
        Z = radius * np.cos(T) + z

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, alpha=0.2)

        sphere_samples = np.array(sphere_samples)
        ax.scatter(sphere_samples[:, 0], sphere_samples[:,
                   1], sphere_samples[:, 2], c='red')

        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Visualization of a sphere and its surface samples")

        # plt.show()

    def gen_poses(self, center, radius, sphere_samples):
        # center coordinates
        xc, yc, zc = center[0], center[1], center[2]
        '''
            RPY sono gli angoli rispetto al sistema di riferimento in cui sono
            stati definiti i centri delle sfere (punti raggiungibili)
        '''
        pose_angles_RPY = []

        for i in range(len(sphere_samples)):
            xs, ys, zs = sphere_samples[i,0], sphere_samples[i, 1], sphere_samples[i, 2]

            # compute the vector parallel to the line going from a sample
            # towards the center of the sphere
            v = [xc-xs, yc-ys, zc-zs]

            # normalize the vector
            v_norm = v / np.linalg.norm(v, ord=2)

            # compute the yaw angle
            yaw = np.arctan2(v_norm[1], v_norm[0])

            # compute the pitch angle
            pitch = -np.arctan2(v_norm[2],
                                np.sqrt(v_norm[1]**2 + v_norm[0]**2))

            # set the roll angle to 0
            roll = 0.

            pose_angles_RPY.append([roll, pitch, yaw])

        return pose_angles_RPY

    def vis_sphere_sample_poses(self, center, radius, sphere_samples, pose_angles_RPY):
        x, y, z = center[0], center[1], center[2]

        alpha = np.linspace(0, 2*np.pi, 50)
        theta = np.linspace(0, np.pi, 50)

        A, T = np.meshgrid(alpha, theta)

        X = radius * np.cos(A) * np.sin(T) + x
        Y = radius * np.sin(A) * np.sin(T) + y
        Z = radius * np.cos(T) + z

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # plot the sphere
        ax.plot_surface(X, Y, Z, alpha=0.2)

        # plot the surface samples
        sphere_samples = np.array(sphere_samples)
        ax.scatter(sphere_samples[:, 0], sphere_samples[:,
                   1], sphere_samples[:, 2], c='red')

        # plot the generated pose
        for i in range(len(pose_angles_RPY)):
            rot_mat = Rotation.from_euler(
                'xyz', pose_angles_RPY[i, :], degrees=False).as_matrix()
            dir = np.array([radius, 0, 0])
            end = np.dot(rot_mat, dir.T)

            ax.quiver(sphere_samples[i, 0], sphere_samples[i, 1], sphere_samples[i, 2],
                      end[0], end[1], end[2])

        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            "Visualization of a sphere, its surface samples and the generated poses")

        # plt.show()

    def create_pointcloud_msg(self, colors):        
        # create the MarkerArray message
        markArray_msg = MarkerArray()
        
        for i in range(self.points.shape[0]):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.type = Marker.SPHERE
            marker.pose.position.x = self.points[i,0]
            marker.pose.position.y = self.points[i,1]
            marker.pose.position.z = self.points[i,2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = colors[i,0]
            marker.color.g = colors[i,1]
            marker.color.b = colors[i,2]
            marker.color.a = 0.7
            
            markArray_msg.markers.append(marker)
        
        return markArray_msg
            
    def publish_pointcloud_msg(self, colors):
        markArray_msg = self.create_pointcloud_msg(colors)
        self.pub_cloud.publish(markArray_msg)
        rospy.sleep(0.5)

if __name__ == "__main__":
    gen_cloud = GenereatePointCloudWithMetric()
    gen_cloud.create_ros_node()
    gen_cloud.create_GUI()

    # gen_cloud.from_extern = True
    # gen_cloud.urdf_file_path = "/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/reach_space_modeling/src/generate_pointcloud/model/mobile_wx250s.urdf"
    # gen_cloud.parse_urdf()
    # gen_cloud.wrist_lst_j_name = "wrist_rotate"
    # gen_cloud.arm_lst_j_name = "elbow"
    # gen_cloud.arm_frt_j_name = "waist"
    # gen_cloud.num_samples = 10
    # gen_cloud.generate_point_cloud()
    rospy.loginfo("Reachability point cloud created...")

    gen_cloud.generate_reachability_index()
    colors = gen_cloud.vis_cloud_with_measure()
    gen_cloud.publish_pointcloud_msg(colors)
    # gen_cloud.vis_cloud_and_poses()
