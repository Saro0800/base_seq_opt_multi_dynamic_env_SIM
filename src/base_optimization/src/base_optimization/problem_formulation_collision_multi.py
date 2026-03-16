#!/usr/bin/env python3
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from scipy.spatial.transform import Rotation


class BasePoseOptProblem(ElementwiseProblem):
    def __init__(self, *args):
        # retrieve the passed arguments
        self.ell_center = args[0]
        self.ell_axis_out = args[1]
        self.ell_axis_inn = args[2]
        self.des_pose_multi = args[3]
        self.des_pose_weigths = args[4]
        self.point_cloud = args[5]
        self.free_space_2d_map = args[6]
        self.occupancy_grid_info = args[7]
        self.ell_center_base = args[8]

        aO = self.ell_axis_out[0]
        bO = self.ell_axis_inn[1]

        # retrieve the points of the occupancy grid
        self.xcloud = np.array([x[0] for x in self.point_cloud])
        self.ycloud = np.array([x[1] for x in self.point_cloud])
        self.zcloud = np.array([x[2] for x in self.point_cloud])

        # define the parameters of the optimization problem
        super().__init__(n_var=3,
                         n_obj=1,
                         n_ieq_constr=4,
                         xl=np.array([min([p.x for p in self.des_pose_multi]) - max(aO, bO),
                                      min([p.y for p in self.des_pose_multi]) - max(aO, bO),
                                      0]),
                         xu=np.array([max([p.x for p in self.des_pose_multi]) + max(aO, bO),
                                      max([p.y for p in self.des_pose_multi]) + max(aO, bO),
                                      360]))

    def _evaluate(self, x, out, *args, **kwargs):
        # retrieve the ellipsoid equation paramters
        aO = self.ell_axis_out[0]
        bO = self.ell_axis_out[1]
        cO = self.ell_axis_out[2]
        
        aI = self.ell_axis_inn[0]
        bI = self.ell_axis_inn[1]
        cI = self.ell_axis_inn[2]
        
        xc = self.ell_center[0]
        yc = self.ell_center[1]
        zc = self.ell_center[2]

        # transformation matrix from R0 to Rell
        matr_R0_Rell = np.zeros((4, 4))
        rot = np.transpose(Rotation.from_euler(
            'xyz', [0., 0., x[2]], degrees=True).as_matrix())
        matr_R0_Rell[:3, :3] = rot
        matr_R0_Rell[:3, 3] = -np.dot(rot, np.array([x[0], x[1], self.ell_center[2]]))
        matr_R0_Rell[3, 3] = 1

        t_in_outEll = []
        t_in_innEll = []
        wt_in_outEll = []
        wt_in_innEll = []
        t_thetas = []    
        all_t_x_pos = 1
        
        # select the desired targets that are ONLY inside the outer ellipsoid 
        for i, t in enumerate(self.des_pose_multi):
            # check if it is inside the inner ellipsoid
            if ((x[0]-t.x)/aI)**2 + ((x[1]-t.y)/bI)**2 + ((zc-t.z)/cI)**2 <=1:
                t_in_innEll.append(t)
                wt_in_innEll.append(self.des_pose_weigths[i])
                
            # check if it is only inside the outer
            elif ((x[0]-t.x)/aO)**2 + ((x[1]-t.y)/bO)**2 + ((zc-t.z)/cO)**2 <=1:
                t_in_outEll.append(t)                    
                wt_in_outEll.append(self.des_pose_weigths[i])
         
        # compute angle for targets that are inside the out Ell only
        for t in t_in_outEll:
            # compute the homog coordinates of each t wrt to Rell
            p_R0 = np.array([t.x, t.y, t.z, 1])
            p_Rell = np.dot(matr_R0_Rell, p_R0)
            angle = np.arctan2(p_Rell[1], p_Rell[0])
            t_thetas.append(angle)
        
            # check if coordinate x of t is negative
            if p_Rell[0] <=0:
                all_t_x_pos = 0
                
        # count how many voxels are inside ellipsoids (collisions).
        # It is enough to count those inside out Ell
        coll_points = self.point_cloud[(
            (x[0]-self.xcloud)/aO)**2 + ((x[1]-self.ycloud)/bO)**2 + ((zc-self.zcloud)/cO)**2 <= 1]
        
        # compute (x,y) position of the base of the robot wrt to map
        homog_matr = np.zeros((4, 4))
        homog_matr[:3, :3] = Rotation.from_euler(
            'xyz', [0, 0, x[2]], degrees=True).as_matrix()
        homog_matr[:3, 3] = np.array([x[0], x[1], 0])
        homog_matr[3, 3] = 1
        base_pos = np.dot(homog_matr, np.array(
            [-self.ell_center_base[0], -self.ell_center_base[1], 0, 1]))
        base_pos[2] = x[2]

        # check if the base position is inside the free space 2D map
        resolution = self.occupancy_grid_info.resolution
        origin_x = self.occupancy_grid_info.origin.position.x
        origin_y = self.occupancy_grid_info.origin.position.y

        robot_pixel_x = int((base_pos[0] - origin_x) / resolution)
        robot_pixel_y = int((base_pos[1] - origin_y) / resolution)
        robot_pixel_y = self.occupancy_grid_info.height - 1 - robot_pixel_y

        # retrieve value of the pixel
        pixel_value = self.free_space_2d_map[robot_pixel_y, robot_pixel_x]
        # print((robot_pixel_x, robot_pixel_y), end=" ")
        if pixel_value > 0:
            # print("Base position is in free space")
            valid_position = 1
        else:
            # print("Base position is NOT in free space")
            valid_position = 0
        
        # define the constraints
        #       pow(10,-6) is necessary because pymoo uses only the
        #       smaller or equal constraints. Since here we need a 
        #       strictly greater constraints, we convert it to a
        #       greater than or equal to constraint of a small quantity
        constraints = [-np.sum(wt_in_outEll) + pow(10,-6),
                       np.sum(wt_in_innEll),
                       -all_t_x_pos + pow(10,-6),
                       -valid_position + pow(10,-6)]
        out["G"] = np.row_stack(constraints)
        
        alpha = 20
        beta = 1000
        gamma = 5
                
        out["F"] =  alpha * coll_points.shape[0]    \
                    - beta * np.sum(wt_in_outEll)         \
                    + gamma * np.abs(np.sum(t_thetas))