#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <ros/ros.h>
#include <string>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "filter_octomap");
    ros::NodeHandle nh("~");

    // Parametri
    std::string input_file, output_file;
    double min_height;

    nh.param<std::string>("input", input_file, "/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/maps/octomap_lab_real.bt");
    nh.param<std::string>("output", output_file, "/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/maps/octomap_lab_real_filtered.bt");
    nh.param<double>("min_height", min_height, 0.05);

    // Carica l'Octomap
    octomap::OcTree* tree = new octomap::OcTree(input_file);
    if (!tree)
    {
        ROS_ERROR("Impossibile caricare il file: %s", input_file.c_str());
        return 1;
    }

    double resolution = tree->getResolution();
    ROS_INFO("File caricato: %s", input_file.c_str());
    ROS_INFO("Risoluzione: %.4f m", resolution);

    // Raccogli le coordinate dei voxel da rimuovere
    std::vector<octomap::point3d> to_remove;

    for (octomap::OcTree::leaf_iterator it = tree->begin_leafs(),
         end = tree->end_leafs(); it != end; ++it)
    {
        if (it.getZ() < min_height)
        {
            to_remove.push_back(it.getCoordinate());
        }
    }

    ROS_INFO("Voxel totali sotto %.2f m: %lu", min_height, to_remove.size());

    // Rimuovi i voxel
    for (const auto& coord : to_remove)
    {
        tree->deleteNode(coord, tree->getTreeDepth());
    }

    // Comprimi l'albero dopo le modifiche
    tree->prune();
    tree->updateInnerOccupancy();

    // Salva
    tree->writeBinary(output_file);
    ROS_INFO("Mappa filtrata salvata in: %s", output_file.c_str());

    delete tree;
    return 0;
}