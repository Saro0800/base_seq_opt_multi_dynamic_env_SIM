#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from segmentation_model import SegmentationModel


class SegmentationNode:
    """Example ROS node using the SegmentationModel class."""
    
    def __init__(self):
        rospy.init_node("segmentation_node")
        rospy.loginfo("Node Created")
        
        # Define the classes you want to detect
        classes_to_detect = ["mobile robot", "person"]
        
        # Instantiate the segmentation model with your desired classes
        self.segmentation_model = SegmentationModel(
            classes=classes_to_detect,
            box_threshold=0.5,
            text_threshold=0.5,
            nms_threshold=0.8
        )
        
        # Create publisher for segmented point cloud
        self.pointcloud_pub = rospy.Publisher(
            "/tmp/segmented_pointcloud", 
            PointCloud2, 
            queue_size=10
        )
        
        rospy.sleep(0.5)
    
    def run(self):
        """Main processing loop."""
        while not rospy.is_shutdown():
            # Request image msg
            img_msg = rospy.wait_for_message("/locobot/camera/color/image_raw", Image)
            
            # Request aligned depth image
            aligned_depth_img = rospy.wait_for_message(
                "/locobot/camera/aligned_depth_to_color/image_raw", 
                Image
            )
            
            aligned_depth_camera_info = rospy.wait_for_message(
                "/locobot/camera/aligned_depth_to_color/camera_info", 
                CameraInfo
            )
            
            # Segment the RGB image
            detections = self.segmentation_model.segment_image(img_msg)
            
            # Apply segmentation to depth image
            pointcloud_msg, points = self.segmentation_model.apply_segmentation_to_depth(
                aligned_depth_camera_info, 
                aligned_depth_img, 
                detections
            )
            
            # Publish the point cloud if valid
            if pointcloud_msg is not None:
                self.pointcloud_pub.publish(pointcloud_msg)
            
            # Optional: get annotated image for visualization
            # annotated_img = self.segmentation_model.get_annotated_image(img_msg, detections)


if __name__ == "__main__":
    try:
        node = SegmentationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass