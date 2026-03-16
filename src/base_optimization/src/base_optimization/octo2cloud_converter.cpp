#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/conversions.h>
#include <ros/ros.h>
#include <base_optimization/octomap2cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>


// function to handle the service request
bool octomap_msg_handler(base_optimization::octomap2cloud::Request  &req,
                         base_optimization::octomap2cloud::Response &res){

    ros::NodeHandle node;
    octomap_msgs::Octomap octo_msg;
    octomap_msgs::GetOctomap srv;
    

    ROS_INFO("Received request to convert the current octomap_binary to a pointcloud");

    ROS_INFO("Requesting the current octomap_binary from '/locobot/octomap_server/octomap_binary'");
    ros::ServiceClient client = node.serviceClient<octomap_msgs::GetOctomap>("/locobot/octomap_server/octomap_binary");

    // send the request (empty)
    if(client.call(srv)){
        // retrieve the response
        ROS_INFO("octomap_binary correctly received");
        octo_msg = srv.response.map;
    }
    else{
        ROS_ERROR("Failed getting the current octomap_binary");
        return false;
    }

    // convert the message into an OcTree
    octomap::AbstractOcTree *abst_octree = octomap_msgs::msgToMap(octo_msg);
    octomap::OcTree *octree = dynamic_cast<octomap::OcTree*>(abst_octree);

    // check for null
    if(!octree){
        ROS_ERROR("Error converting the AbstractOcTree to a OcTree");
        return false;
    }

    // create a pcl pointcloud
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // iterate on all leaves (voxels) building the octree
    for(octomap::OcTree::leaf_iterator it = octree->begin(), end = octree->end(); it != end; ++it)
        // check if the node is occupied or not
        if(octree->isNodeOccupied(*it)){
            cloud.push_back(pcl::PointXYZ(it.getX(), it.getY(), it.getZ()));
            // ROS_INFO("point %.2f, %.2f, %.2f\n", it.getX(), it.getY(), it.getZ());
        }
    ROS_INFO("Occupancy map converted into a point cloud");

    // convert the pcl point cloud into a ROS message (PointCloud2)
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";
    cloud_msg.header.stamp = ros::Time::now();
    ROS_INFO("Created the PointCloud2 message");

    // crate the service response
    res.cloud = cloud_msg;  
    ROS_INFO("Created the service response");


    return true;
}

int main(int argc, char** argv) {
    // init the ROS node
    ros::init(argc, argv, "octo2cloud_converter_node");
    ros::NodeHandle node;
    ROS_INFO("Node 'octo2cloud_converter_node' created");

    /* 
     * create a service to request the conversion between
     * an OctoMap (binary) and a PointCloud2
     */
    ros::ServiceServer octo2cloud_server = node.advertiseService(
                                                "octomap2cloud_converter_srv",
                                                octomap_msg_handler);
    ROS_INFO("Node 'octo2cloud_converter_node' created service 'octomap2cloud_converter_srv'");


    // start spinning
    ROS_INFO("Node 'octo2cloud_converter_node' starts spinning ...");
    ros::spin();

    return 0;
}
