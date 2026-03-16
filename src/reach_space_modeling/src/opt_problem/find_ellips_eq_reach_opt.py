import numpy as np
import rospy
import math
import time

from opt_problem.problem_formulation_reach_opt import EllipsoidEquationOptProblem
from generate_pointcloud.gen_cloud_reach_metric import GenereatePointCloudWithMetric

from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from sensor_msgs.point_cloud2 import create_cloud
from reach_space_modeling.srv import ell_params, ell_paramsRequest, ell_paramsResponse
from gazebo_msgs.srv import GetPhysicsProperties

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.path import Path
import mpl_toolkits.mplot3d.art3d as art3d

res = None


def solve_eqn_prob(points, pnt_weights, alg_name, link=None, center=None, viz_res=False):
    # define the problem
    problem = EllipsoidEquationOptProblem(
        center, points, pnt_weights, viz_res, None, None)

    # define the weights of the points and of the volume, and the optimization algorithm
    problem.num_points_wt = 1
    problem.volume_wt = pow(10, int(math.log10(points.shape[0])-1))

    x0 = np.array([[1, 1, 1,
                    0.1, 0.1, 0.1,
                    problem.center[0], problem.center[1], problem.center[2]]])

    if alg_name == "GA":
        pop = Population.new(X=x0)
        algorithm = GA(sampling=pop)

    elif alg_name == "PSO":
        # problem.volume_wt = 1
        init_pop = np.repeat(x0, 25, axis=0)
        algorithm = PSO(sampling=init_pop, adaptive=True, pertube_best=False)

    termination = RobustTermination(
        SingleObjectiveSpaceTermination(tol=pow(10, -6))
    )

    # solve the optimization problem
    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=termination,
                   verbose=False,
                   seed=1)

    if viz_res == True:
        print("Best solution found: \n\ta={:.4f}, b={:.4f}, c={:.4f}\n\txC={:.4f}, yC={:.4f}, zC={:.4f}".format(
            res.X[0], res.X[1], res.X[2], res.X[3], res.X[4], res.X[5]))

    print(res.X)

    print(res.F)

    return res


def create_ell_msg(points, link, axes, center=None):
    # retrieve the solution of the opt problem
    a = axes[0]
    b = axes[1]
    c = axes[2]

    # compute the center
    if center is None:
        center = np.array([np.mean([np.min(points[:, 0]), np.max(points[:, 0])]),
                           np.mean([np.min(points[:, 1]),
                                   np.max(points[:, 1])]),
                           np.mean([np.min(points[:, 2]), np.max(points[:, 2])])])

    marker_msg = Marker()
    marker_msg.header.frame_id = link
    marker_msg.header.stamp = rospy.Time.now()

    marker_msg.frame_locked = True

    # set the shape
    marker_msg.type = 2
    marker_msg.id = 0

    # set the scale of the marker
    marker_msg.scale.x = 2*a
    marker_msg.scale.y = 2*b
    marker_msg.scale.z = 2*c

    # set the color
    marker_msg.color.r = 100/255
    marker_msg.color.g = 100/255
    marker_msg.color.b = 100/255
    marker_msg.color.a = 0.2

    # set the pose of the marker
    marker_msg.pose.position.x = center[0]
    marker_msg.pose.position.y = center[1]
    marker_msg.pose.position.z = center[2]
    marker_msg.pose.orientation.x = 0.0
    marker_msg.pose.orientation.y = 0.0
    marker_msg.pose.orientation.z = 0.0
    marker_msg.pose.orientation.w = 1.0

    return marker_msg


def create_cloud_msg(points, link):
    marker_msg = Marker()
    marker_msg.header.frame_id = link
    marker_msg.header.stamp = rospy.Time.now()

    marker_msg.frame_locked = True

    # set the shape
    marker_msg.type = 7
    marker_msg.action = marker_msg.ADD
    marker_msg.id = 0

    # set the scale of the marker
    marker_msg.scale.x = 0.01
    marker_msg.scale.y = 0.01
    marker_msg.scale.z = 0.01

    marker_msg.pose.orientation.x = 0.0
    marker_msg.pose.orientation.y = 0.0
    marker_msg.pose.orientation.z = 0.0
    marker_msg.pose.orientation.w = 1.0

    for p in points:
        # add a point
        marker_msg.points.append(Point(p[0], p[1], p[2]))
        # add a color for the point
        marker_msg.colors.append(ColorRGBA(1.0, 0.0, 0.0, 0.2))

    return marker_msg


def give_ell_params(req):
    aO = res.X[0]
    bO = res.X[1]
    cO = res.X[2]

    aI = res.X[3]
    bI = res.X[4]
    cI = res.X[5]

    xC = res.X[6]
    yC = res.X[7]
    zC = res.X[8]
    ell_ref_frame = link

    return aO, bO, cO, aI, bI, cI, xC, yC, zC, ell_ref_frame


def vis_2d_opt_RS(points, reach_meas, center, out_p, inn_p):
    colors = ['red', 'yellow', 'lightgreen', 'blue']  # low to high
    custom_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
    tol = 0.05
    
    # XY Plane Projection
    sec_points = [p for p in points if (p[2] >= center[2]-tol) and (p[2] <= center[2]+tol)]
    sec_points = np.array(sec_points)
    sec_reach_mes = [reach_meas[i] for i, p in enumerate(points) if (p[2] >= center[2]-tol) and (p[2] <= center[2]+tol)]
    sec_reach_mes = np.array(sec_reach_mes)
    
    axes[0].set_facecolor('w')
    sc = axes[0].scatter(sec_points[:, 0], sec_points[:, 1], s=10,
                        c=sec_reach_mes, cmap=custom_cmap, alpha=1)
    
    # Create smooth ellipse paths manually
    theta = np.linspace(0, 2*np.pi, 200)  # 200 points for smooth ellipse
    
    # Outer ellipse
    x_out = center[0] + out_p[0] * np.cos(theta)
    y_out = center[1] + out_p[1] * np.sin(theta)
    
    # Inner ellipse (reversed for proper path direction)
    x_inn = center[0] + inn_p[0] * np.cos(theta[::-1])
    y_inn = center[1] + inn_p[1] * np.sin(theta[::-1])
    
    # Combine vertices
    vertices = np.vstack([
        np.column_stack([x_out, y_out]),
        np.column_stack([x_inn, y_inn])
    ])
    
    # Create path codes
    codes = np.full(len(x_out), Path.LINETO)
    codes[0] = Path.MOVETO
    inner_codes = np.full(len(x_inn), Path.LINETO)
    inner_codes[0] = Path.MOVETO
    all_codes = np.concatenate([codes, inner_codes])
    
    # Create path and patch
    path = Path(vertices, all_codes)
    area = mpatches.PathPatch(path, linewidth=1.5,
                              edgecolor=(100/255, 100/255, 100/255), 
                              facecolor=(100/255, 100/255, 100/255, 0.2))
    axes[0].add_patch(area)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y', labelpad=10)
    axes[0].axis("equal")

    # XZ Plane Projection (similar approach)
    sec_points = [p for p in points if (p[1] >= center[1]-tol) and (p[1] <= center[1]+tol)]
    sec_points = np.array(sec_points)
    sec_reach_mes = [reach_meas[i] for i, p in enumerate(points) if (p[1] >= center[1]-tol) and (p[1] <= center[1]+tol)]
    sec_reach_mes = np.array(sec_reach_mes)
    
    axes[1].set_facecolor('w')
    sc = axes[1].scatter(sec_points[:, 0], sec_points[:, 2], s=10,
                        c=sec_reach_mes, cmap=custom_cmap, alpha=1)
    
    # Create smooth ellipse paths manually
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Outer ellipse
    x_out = center[0] + out_p[0] * np.cos(theta)
    z_out = center[2] + out_p[2] * np.sin(theta)
    
    # Inner ellipse (reversed)
    x_inn = center[0] + inn_p[0] * np.cos(theta[::-1])
    z_inn = center[2] + inn_p[2] * np.sin(theta[::-1])
    
    # Combine vertices
    vertices = np.vstack([
        np.column_stack([x_out, z_out]),
        np.column_stack([x_inn, z_inn])
    ])
    
    # Create path codes
    codes = np.full(len(x_out), Path.LINETO)
    codes[0] = Path.MOVETO
    inner_codes = np.full(len(x_inn), Path.LINETO)
    inner_codes[0] = Path.MOVETO
    all_codes = np.concatenate([codes, inner_codes])
    
    # Create path and patch
    path = Path(vertices, all_codes)
    area = mpatches.PathPatch(path, linewidth=1.5,
                              edgecolor=(100/255, 100/255, 100/255),
                              facecolor=(100/255, 100/255, 100/255, 0.2))
    axes[1].add_patch(area)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z', labelpad=10)
    axes[1].axis("equal")

    # YZ Plane Projection (similar approach)
    sec_points = [p for p in points if (p[0] >= center[0]-tol) and (p[0] <= center[0]+tol)]
    sec_points = np.array(sec_points)
    sec_reach_mes = [reach_meas[i] for i, p in enumerate(points) if (p[0] >= center[0]-tol) and (p[0] <= center[0]+tol)]
    sec_reach_mes = np.array(sec_reach_mes)
    
    axes[2].set_facecolor('w')
    sc = axes[2].scatter(sec_points[:, 1], sec_points[:, 2], s=10,
                        c=sec_reach_mes, cmap=custom_cmap, alpha=1)
    
    # Create smooth ellipse paths manually
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Outer ellipse
    y_out = center[1] + out_p[1] * np.cos(theta)
    z_out = center[2] + out_p[2] * np.sin(theta)
    
    # Inner ellipse (reversed)
    y_inn = center[1] + inn_p[1] * np.cos(theta[::-1])
    z_inn = center[2] + inn_p[2] * np.sin(theta[::-1])
    
    # Combine vertices
    vertices = np.vstack([
        np.column_stack([y_out, z_out]),
        np.column_stack([y_inn, z_inn])
    ])
    
    # Create path codes
    codes = np.full(len(y_out), Path.LINETO)
    codes[0] = Path.MOVETO
    inner_codes = np.full(len(y_inn), Path.LINETO)
    inner_codes[0] = Path.MOVETO
    all_codes = np.concatenate([codes, inner_codes])
    
    # Create path and patch
    path = Path(vertices, all_codes)
    area = mpatches.PathPatch(path, linewidth=1.5,
                              edgecolor=(100/255, 100/255, 100/255), 
                              facecolor=(100/255, 100/255, 100/255, 0.2))
    axes[2].add_patch(area)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z', labelpad=10)
    axes[2].axis("equal")

    # cbar = fig.colorbar(sc, ax=axes, shrink=0.8,
    #                    location="top", orientation="horizontal")
    # cbar.set_label("Points reachability measure", fontsize=12)

    # plt.show()


def vis_3d_opt_RS(points, reach_meas, center, out_p, inn_p):
    colors = ['red', 'yellow', 'lightgreen', 'blue']  # low to high
    custom_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('w')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # draw the point cloud with the reachability measure as colour
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=reach_meas, cmap="plasma_r", s=0, alpha=1)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, location="left")
    cbar.set_label("Points reachability measure", fontsize=12)
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=reach_meas, cmap="plasma_r", s=10, alpha=0.1)

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)

    # add the outer ellipsoid
    xO = center[0] + out_p[0]*np.sin(V)*np.cos(U)
    yO = center[1] + out_p[1]*np.sin(V)*np.sin(U)
    zO = center[2] + out_p[2]*np.cos(V)
    ax.plot_surface(xO, yO, zO, alpha=0.4, color=(0, 0.5, 0))

    # add the inner ellipsoid
    xI = center[0] + inn_p[0]*np.sin(V)*np.cos(U)
    yI = center[1] + inn_p[1]*np.sin(V)*np.sin(U)
    zI = center[2] + inn_p[2]*np.cos(V)
    ax.plot_surface(xI, yI, zI, alpha=1, color=(0, 0.5, 0))

    tol = 0.07
    # add the projection of the outer ellipsoid on a plane parallel
    # to the xy plane and passing from the center.
    sec_points = [p
                  for p in points
                  if (p[2] >= center[2]-tol) and (p[2] <= center[2]+tol)]
    sec_points = np.array(sec_points)
    sec_reach_mes = [reach_meas[i]
                     for i, p in enumerate(points)
                     if (p[2] >= center[2]-tol) and (p[2] <= center[2]+tol)]
    sec_reach_mes = np.array(sec_reach_mes)
    sc = ax.scatter(sec_points[:, 0], sec_points[:, 1], center[2]-1.1,
                    c=sec_reach_mes, cmap=custom_cmap, s=10, alpha=1)
    ell_out = Ellipse((center[0], center[1]), 2*out_p[0], 2*out_p[1],
                      fill=False, linewidth=1.5, edgecolor=(0, 0.5, 0))
    ell_inn = Ellipse((center[0], center[1]), 2*inn_p[0], 2*inn_p[1],
                      fill=False, linewidth=1.5, edgecolor=(0, 0.5, 0))
    ax.add_patch(ell_out)
    ax.add_patch(ell_inn)
    art3d.pathpatch_2d_to_3d(ell_out, z=center[2]-1.1, zdir='z')
    art3d.pathpatch_2d_to_3d(ell_inn, z=center[2]-1.1, zdir='z')

    # add the projection of the outer ellipsoid on a plane parallel
    # to the xz plane and passing from the center.
    sec_points = [p
                  for p in points
                  if (p[1] >= center[1]-tol) and (p[1] <= center[1]+tol)]
    sec_points = np.array(sec_points)
    sec_reach_mes = [reach_meas[i]
                     for i, p in enumerate(points)
                     if (p[1] >= center[1]-tol) and (p[1] <= center[1]+tol)]
    sec_reach_mes = np.array(sec_reach_mes)
    sc = ax.scatter(sec_points[:, 0], center[1]-1.1, sec_points[:, 2],
                    c=sec_reach_mes, cmap=custom_cmap, s=10, alpha=1)
    ell_out = Ellipse((center[0], center[2]), 2*out_p[0], 2*out_p[2],
                      fill=False, linewidth=1.5, edgecolor=(0, 0.5, 0))
    ell_inn = Ellipse((center[0], center[2]), 2*inn_p[0], 2*inn_p[2],
                      fill=False, linewidth=1.5, edgecolor=(0, 0.5, 0))
    ax.add_patch(ell_out)
    ax.add_patch(ell_inn)
    art3d.pathpatch_2d_to_3d(ell_out, z=center[1]-1.1, zdir='y')
    art3d.pathpatch_2d_to_3d(ell_inn, z=center[1]-1.1, zdir='y')

    # add the projection of the outer ellipsoid on a plane parallel
    # to the yz plane and passing from the center.
    sec_points = [p
                  for p in points
                  if (p[0] >= center[0]-tol) and (p[0] <= center[0]+tol)]
    sec_points = np.array(sec_points)
    sec_reach_mes = [reach_meas[i]
                     for i, p in enumerate(points)
                     if (p[0] >= center[0]-tol) and (p[0] <= center[0]+tol)]
    sec_reach_mes = np.array(sec_reach_mes)
    sc = ax.scatter(center[0]-1.1, sec_points[:, 1], sec_points[:, 2],
                    c=sec_reach_mes, cmap=custom_cmap, s=10, alpha=1)
    ell_out = Ellipse((center[1], center[2]), 2*out_p[1], 2*out_p[2],
                      fill=False, linewidth=1.5, edgecolor=(0, 0.5, 0))
    ell_inn = Ellipse((center[1], center[2]), 2*inn_p[1], 2*inn_p[2],
                      fill=False, linewidth=1.5, edgecolor=(0, 0.5, 0))
    ax.add_patch(ell_out)
    ax.add_patch(ell_inn)
    art3d.pathpatch_2d_to_3d(ell_out, z=center[0]-1.1, zdir='x')
    art3d.pathpatch_2d_to_3d(ell_inn, z=center[0]-1.1, zdir='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis("equal")
    ax.view_init(elev=39, azim=44)
    # plt.show()


if __name__ == "__main__":
    # get the pointcloud points
    gen_cloud = GenereatePointCloudWithMetric()
    gen_cloud.create_ros_node()

    # # wait for gazebo to be unpaued
    # rospy.wait_for_service("/gazebo/get_physics_properties")

    # get_physics = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)

    # rospy.loginfo("Waiting for Gazebo to be unpaused...")

    # while not rospy.is_shutdown():
    #     try:
    #         physics = get_physics()
    #         if physics.pause != True:  # Gazebo is unpaused if gravity is nonzero
    #             rospy.loginfo("Gazebo unpaused! Proceeding...")
    #             break
    #     except rospy.ServiceException:
    #         pass  # If service is not available, keep trying

    #     rospy.sleep(1)

    # gen_cloud.create_GUI()

    # generate the point cloud
    gen_cloud.from_extern = True
    gen_cloud.urdf_file_path = "/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/reach_space_modeling/src/generate_pointcloud/model/mobile_wx250s.urdf"
    gen_cloud.parse_urdf()
    gen_cloud.wrist_lst_j_name = "wrist_rotate"
    gen_cloud.arm_lst_j_name = "elbow"
    gen_cloud.arm_frt_j_name = "waist"
    gen_cloud.num_samples = 10
    gen_cloud.generate_point_cloud()
    rospy.loginfo("Reachability point cloud created...")

    # compute the reachability index for each point
    gen_cloud.generate_reachability_index()
    # gen_cloud.vis_cloud_with_measure()

    # count how many points with a given reach measure are available
    freq = np.zeros(int(np.max(gen_cloud.points_reach_measure)+1), dtype=int)
    for i in range(gen_cloud.points_reach_measure.shape[0]):
        freq[int(gen_cloud.points_reach_measure[i])] = freq[int(
            gen_cloud.points_reach_measure[i])] + 1

    for i in range(freq.shape[0]):
        print("Points reachable with {:d} poses: {:d}".format(i, freq[i]))

    # weighted mean of the reachability index
    w_mean = 0
    for i in range(freq.shape[0]):
        w_mean = w_mean + freq[int(i)]*float(i)
    w_mean = w_mean/gen_cloud.points.shape[0]
    print(
        "\nWeighted mean of all reachability measures: {:.2f}\n".format(w_mean))

    # define a weight for each point based on its reachability measure
    pnt_weights = np.zeros(gen_cloud.points.shape[0])
    for i in range(gen_cloud.points.shape[0]):
        pnt_weights[i] = int(w_mean) - gen_cloud.points_reach_measure[i]

    # solve the optimization problem
    points = gen_cloud.points
    link = gen_cloud.point_cloud_orig_frame
    center = np.array([np.mean([np.min(points[:, 0]), np.max(points[:, 0])]),
                       np.mean([np.min(points[:, 1]), np.max(points[:, 1])]),
                       np.mean([np.min(points[:, 2]), np.max(points[:, 2])])])

    start = time.time()

    alg_name = "GA"
    # alg_name = "PSO"
    res = solve_eqn_prob(points, pnt_weights, alg_name,
                         link, center, viz_res=False)

    # res.X[0:3] = res.X[0:3] * 0.8
    # res.X[6] = res.X[6] - 0.1
    # res.X[8] = res.X[8] - 0.1
    out_p = res.X[0:3]
    inn_p = res.X[3:6]
    center = res.X[6:9]

    vis_2d_opt_RS(points, gen_cloud.points_reach_measure, center, out_p, inn_p)
    vis_3d_opt_RS(points, gen_cloud.points_reach_measure, center, out_p, inn_p)

    end = time.time() - start
    print("Solution found in {:.4f}s".format(end))

    # create the point cloud message
    cloud_msg = create_cloud_msg(
        gen_cloud.points, gen_cloud.point_cloud_orig_frame)

    # create the center of the point cloud message
    center_msg = create_cloud_msg([center], gen_cloud.point_cloud_orig_frame)

    # create the outer ellipsoid message
    ell_out_msg = create_ell_msg(
        None, gen_cloud.point_cloud_orig_frame, res.X[:3], center)

    # create the inner ellipsoid message
    ell_inn_msg = create_ell_msg(
        None, gen_cloud.point_cloud_orig_frame, res.X[3:6], center)

    # provide a service to  get the parameters of the ellipsoid
    ell_params_srv = rospy.Service(
        "get_ellipsoid_params", ell_params, give_ell_params)

    pub_rate = rospy.Rate(0.5)  # 1 Hz

    # publish the point cloud
    orig_points = gen_cloud.points
    orig_reach_measure = gen_cloud.points_reach_measure
    
    points = gen_cloud.points[gen_cloud.points[:,0]<=center[0]]
    reach_measure = gen_cloud.points_reach_measure[gen_cloud.points[:,0]<=center[0]]
    
    gen_cloud.points = points
    gen_cloud.points_reach_measure = reach_measure
    
    colors = gen_cloud.vis_cloud_with_measure()
    markArray_msg = gen_cloud.create_pointcloud_msg(colors)
    
    print("Start publishing message")
    # for i in range(10):
    while not rospy.is_shutdown():
        gen_cloud.pub_cloud.publish(markArray_msg)
        pub_rate.sleep()

        # publish the center of the point cloud
        center_msg.header.stamp = rospy.Time.now()
        gen_cloud.pub_center.publish(center_msg)
        pub_rate.sleep()

        # publish the outer ellipsoid
        ell_out_msg.header.stamp = rospy.Time.now()
        gen_cloud.pub_ellipsoid_out.publish(ell_out_msg)
        pub_rate.sleep()

        # publish the inner ellipsoid
        ell_inn_msg.header.stamp = rospy.Time.now()
        gen_cloud.pub_ellipsoid_inn.publish(ell_inn_msg)
        pub_rate.sleep()

    rospy.spin()
