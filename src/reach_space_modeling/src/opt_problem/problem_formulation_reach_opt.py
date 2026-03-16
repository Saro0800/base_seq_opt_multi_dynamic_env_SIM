#!/usr/bin/env python3
import sys
sys.path.append("..")

import numpy as np
import math
from pymoo.core.problem import ElementwiseProblem

class EllipsoidEquationOptProblem(ElementwiseProblem):
    def __init__(self, *args):
        # retrieve the set of points
        self.center = args[0]
        self.points = args[1]
        self.pnt_weights = args[2]
        self.num_points = self.points.shape[0]
        self.viz_res = args[3]
        self.num_points_wt = args[4]
        self.volume_wt = args[5]

        # retrieve the passed center
        if self.center is None:
            # compute an estimation of the center
            xc = np.mean([np.min(self.points[:,0]), np.max(self.points[:,0])])
            yc = np.mean([np.min(self.points[:,1]), np.max(self.points[:,1])])
            zc = np.mean([np.min(self.points[:,2]), np.max(self.points[:,2])])
            self.center = np.array([xc, yc, zc])

        if self.viz_res==True:
            print("Center estimation: {:.4f}, {:.4f}, {:.4f}".format(self.center[0], self.center[1], self.center[2]))

        super().__init__(n_var = 9,
                         n_obj = 1,
                         n_ieq_constr = 6,
                         xl = np.concatenate((pow(10,-4)*np.ones(6), -10*np.ones(3))),
                         xu = 10)

    def _evaluate(self, x, out, *args, **kwargs):
        # retrieve the length of the axis of the outer ellipsoid
        aO = x[0]
        bO = x[1]
        cO = x[2]
        
        # retrieve the length of the axis of the inner ellipsoid
        aI = x[3]
        bI = x[4]
        cI = x[5]

        # retrieve the approximation of the center
        xc = x[6]
        yc = x[7]
        zc = x[8]

        # retrieve the points composing the pointcloud
        points = self.points
        xp, yp, zp = points[:,0], points[:,1], points[:,2]
        
        # check wich points are actually inside the outer equation
        mask = ((xp-xc)**2)/(aO**2) + ((yp-yc)**2)/(bO**2) + ((zp-zc)**2)/(cO**2) - 1
        outer_wgt = np.sum(self.pnt_weights[mask<=0])
        outer_pnts = points[mask<=0]
        outer_pnt_weights = self.pnt_weights[mask<=0]
        
        # among those inside the outer ellipsoid, selecte the worst weight
        xp, yp, zp = outer_pnts[:,0], outer_pnts[:,1], outer_pnts[:,2]
        mask = ((xp-xc)**2)/(aI**2) + ((yp-yc)**2)/(bI**2) + ((zp-zc)**2)/(cI**2) - 1
        inner_wgt = np.sum(outer_pnt_weights[mask<=0])
        
        # compute the volumes
        outer_vol = 4/3*np.pi*aO*bO*cO
        inner_vol = 4/3*np.pi*aI*bI*cI
        
        
        # constraints definition
        constrs = [-x[0], -x[1], -x[2], x[3]-x[0], x[4]-x[1], x[5]-x[2]]
        out["G"] = np.row_stack(constrs)
    
        # objective function definition
        # out["F"] = self.num_points_wt*outer_wgt + self.volume_wt*4/3*np.pi*aO*bO*cO -self.num_points_wt*inner_wgt + self.volume_wt*4/3*np.pi*aI*bI*cI
        out["F"] = self.num_points_wt*(outer_wgt - inner_wgt) + self.volume_wt*(outer_vol + inner_vol)