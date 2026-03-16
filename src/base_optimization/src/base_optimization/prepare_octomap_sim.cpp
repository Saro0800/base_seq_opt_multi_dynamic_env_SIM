/**
 * prepare_octomap_sim
 *
 * Pre-processes a .bt octomap generated from mesh voxelization (binvox)
 * so that it works correctly with octomap_server's projected_map.
 *
 * The problem: binvox only marks surface voxels as occupied.
 *   - Floor voxels (z ≈ 0) are occupied → projected_map shows floor as obstacle
 *   - Everything else is "unknown" → projected_map shows unknown instead of free
 *
 * This tool:
 *   1. Marks floor voxels (z < ground_height) as FREE
 *   2. Fills all unknown voxels within the bounding box as FREE
 *   3. Saves the fixed .bt
 *
 * After running, octomap_server can use occupancy_min_z=0 and the
 * projected_map will correctly show free space everywhere except real obstacles.
 *
 * Usage:
 *   rosrun base_optimization prepare_octomap_sim_node \
 *       _input:=/path/to/original.bt \
 *       _output:=/path/to/fixed.bt \
 *       _ground_height:=0.03 \
 *       _fill_unknown:=true
 */

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <ros/ros.h>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "prepare_octomap_sim");
    ros::NodeHandle nh("~");

    // ---- Parameters --------------------------------------------------------
    std::string input_file, output_file;
    double ground_height;
    bool fill_unknown;

    nh.param<std::string>("input",  input_file,
        "/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/maps/v_lab_octomap.bt");
    nh.param<std::string>("output", output_file,
        "/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/maps/v_lab_octomap_sim.bt");
    nh.param<double>("ground_height", ground_height, 0.03);
    nh.param<bool>("fill_unknown", fill_unknown, true);

    // ---- Load the octree ---------------------------------------------------
    octomap::OcTree* tree = new octomap::OcTree(input_file);
    if (!tree)
    {
        ROS_ERROR("Cannot load file: %s", input_file.c_str());
        return 1;
    }

    double res = tree->getResolution();
    ROS_INFO("Loaded: %s  (resolution %.4f m)", input_file.c_str(), res);

    // ---- Save the FULL bounding box BEFORE any modification ----------------
    //  getMetricMin/Max only considers existing leaves, so we must capture the
    //  extent now — before deleting floor voxels — to ensure the fill pass
    //  covers the entire original footprint.
    double bb_x_min, bb_y_min, bb_z_min, bb_x_max, bb_y_max, bb_z_max;
    tree->getMetricMin(bb_x_min, bb_y_min, bb_z_min);
    tree->getMetricMax(bb_x_max, bb_y_max, bb_z_max);
    ROS_INFO("Original bounding box: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)",
             bb_x_min, bb_y_min, bb_z_min, bb_x_max, bb_y_max, bb_z_max);

    // ---- Step 1: DELETE floor voxels ----------------------------------------
    //  Collect coordinates first, then delete (iterating while modifying is
    //  unsafe in octomap).  We delete rather than marking free so that no
    //  residual log-odds remain — the fill_unknown pass will then set them
    //  cleanly to free.
    std::vector<octomap::point3d> floor_coords;

    for (octomap::OcTree::leaf_iterator it = tree->begin_leafs(),
         end = tree->end_leafs(); it != end; ++it)
    {
        if (it.getZ() < ground_height)
        {
            floor_coords.push_back(it.getCoordinate());
        }
    }

    ROS_INFO("Floor voxels to delete (z < %.3f): %lu",
             ground_height, floor_coords.size());

    for (const auto& coord : floor_coords)
    {
        tree->deleteNode(coord, tree->getTreeDepth());
    }

    // ---- Step 2: Fill ALL unknown voxels as free within original bbox ------
    if (fill_unknown)
    {
        ROS_INFO("Filling unknown voxels as free within original bounding box...");

        // Use the clamping min threshold as log-odds value → guaranteed free
        float free_logodds = tree->getClampingThresMinLog();

        double half_res = res * 0.5;
        long filled = 0;

        for (double x = bb_x_min; x <= bb_x_max + half_res; x += res)
        {
            for (double y = bb_y_min; y <= bb_y_max + half_res; y += res)
            {
                for (double z = bb_z_min; z <= bb_z_max + half_res; z += res)
                {
                    octomap::OcTreeNode* node = tree->search(x, y, z);
                    if (node == NULL)
                    {
                        // Unknown voxel → force to free
                        node = tree->updateNode(
                            octomap::point3d(x, y, z), true);
                        node->setLogOdds(free_logodds);
                        filled++;
                    }
                }
            }
        }

        ROS_INFO("Unknown voxels filled as free: %ld", filled);
    }

    // ---- Finalise and save -------------------------------------------------
    tree->updateInnerOccupancy();
    tree->prune();

    tree->writeBinary(output_file);
    ROS_INFO("Saved fixed octomap to: %s", output_file.c_str());

    delete tree;
    return 0;
}
