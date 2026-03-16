import os
import cv2
import numpy as np
import supervision as sv
from cv_bridge import CvBridge

import torch
import torchvision

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid

import prova_pkg_grounded_sam.groundingdino as GroundingDINO
import prova_pkg_grounded_sam.segment_anything as SAM
import prova_pkg_grounded_sam.LightHQSAM as LightHQSAM

from prova_pkg_grounded_sam.groundingdino.util.inference import Model
from prova_pkg_grounded_sam.segment_anything import SamPredictor
from prova_pkg_grounded_sam.LightHQSAM.setup_light_hqsam import setup_model


class SegmentationNode:
    def __init__(self):
        """Initialize the segmentation node with models and ROS components"""
        rospy.loginfo("Initializing SegmentationNode...")
        
        # CvBridge instance
        self.bridge = CvBridge()
        
        # Device for inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        # Detection parameters
        self.classes = ["black robot with a robotic arm", "person"]
        self.box_threshold = 0.7
        self.text_threshold = 0.7
        self.nms_threshold = 0.8
        
        # Initialize GroundingDINO
        grounding_dino_folder = os.path.dirname(os.path.abspath(GroundingDINO.__file__))
        grounding_dino_config_path = os.path.join(grounding_dino_folder, "config/GroundingDINO_SwinT_OGC.py")
        grounding_dino_checkpoint_path = os.path.join(grounding_dino_folder, "config/groundingdino_swint_ogc.pth")
        
        rospy.loginfo("Loading GroundingDINO model...")
        self.grounding_dino_model = Model(
            model_config_path=grounding_dino_config_path,
            model_checkpoint_path=grounding_dino_checkpoint_path
        )
        
        # Initialize LightHQSAM
        hqsam_folder = os.path.dirname(os.path.abspath(LightHQSAM.__file__))
        hqsam_checkpoint_path = os.path.join(hqsam_folder, "config/sam_hq_vit_tiny.pth")
        
        rospy.loginfo("Loading LightHQSAM model...")
        checkpoint = torch.load(hqsam_checkpoint_path)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=self.device)
        self.sam_predictor = SamPredictor(light_hqsam)
        
        # Camera info (cached once received)
        self.camera_info = None
        
        # Octomap projected_map (cached latest received)
        self.octomap_projected_map = None
        self.octomap_map_info = None  # Store map metadata (resolution, origin, etc.)
        self.octomap_projected_map_img = None  # Static background image from first projected_map
        
        # TF2 buffer and listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # ROS Publishers
        self.pointcloud_pub = rospy.Publisher("/segmented_pointcloud", PointCloud2, queue_size=10)
        self.projected_img_pub = rospy.Publisher("/segmented_projected_map", Image, queue_size=10)
        self.segmented_img_pub = rospy.Publisher("/segmented_image", Image, queue_size=1)
        rospy.loginfo("Publishers created")
        
        # Subscribe to octomap projected_map
        rospy.loginfo("Waiting for octomap projected_map...")
        self.octomap_projected_map_msg = rospy.wait_for_message("/locobot/octomap_server/projected_map", OccupancyGrid)
        self.octomap_projected_map_img = self.occupancy_grid_to_image(self.octomap_projected_map_msg)
        rospy.loginfo("Octomap projected_map received!")
        
        # Subscribe to camera info
        rospy.Subscriber("/locobot/camera/aligned_depth_to_color/camera_info", 
                        CameraInfo, self.camera_info_callback, queue_size=1)
        
        # Wait for camera info to be received
        rospy.loginfo("Waiting for camera info...")
        while self.camera_info is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        # Create synchronized subscribers for RGB and Depth
        rgb_sub = Subscriber("/locobot/camera/color/image_raw", Image, queue_size=1)
        depth_sub = Subscriber("/locobot/camera/aligned_depth_to_color/image_raw", Image, queue_size=1)
        
        # Synchronize messages with approximate time sync
        ats = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 
            queue_size=1,
            slop=0.05
        )
        ats.registerCallback(self.synchronized_callback)
        
        rospy.loginfo("SegmentationNode initialization complete!")
    
    def occupancy_grid_to_image(self, grid_msg):
        """Convert a nav_msgs/OccupancyGrid to a BGR image (uint8).

        Mapping:
        -1  (unknown)  → grey  (127, 127, 127)
        0  (free)     → white (255, 255, 255)
        100  (occupied) → black (  0,   0,   0)
        Values in between are linearly interpolated.
        """
        w = grid_msg.info.width
        h = grid_msg.info.height
        
        self.octomap_map_info = {
                'width': w,
                'height': h,
                'frame_id': grid_msg.header.frame_id,
                'resolution': grid_msg.info.resolution,
                'origin_x': grid_msg.info.origin.position.x,
                'origin_y': grid_msg.info.origin.position.y,
        }
        
        data = np.array(grid_msg.data, dtype=np.int8).reshape((h, w))

        img = np.full((h, w, 3), 127, dtype=np.uint8)  # default: unknown → grey

        # Known cells
        known_mask = data >= 0
        # Map 0→255 (free/white) and 100→0 (occupied/black)
        vals = 255 - (data[known_mask].astype(np.float32) * 255.0 / 100.0)
        vals = np.clip(vals, 0, 255).astype(np.uint8)
        img[known_mask, 0] = vals
        img[known_mask, 1] = vals
        img[known_mask, 2] = vals

        # Flip vertically so that Y-up matches image coordinates
        img = cv2.flip(img, 0)
        
        return img
    
    def create_pointcloud2(self, points, header, frame_id):
        """Create a PointCloud2 message from 3D points"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        
        pc2_msg = pc2.create_cloud(header, fields, points)
        return pc2_msg
    
    def segment_image(self, img_msg):
        """Segment the RGB image using GroundingDINO and SAM"""
        # Convert ROS message to cv image
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=cv_img,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # NMS post process
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.nms_threshold
        ).numpy().tolist()
        
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        
        # Early exit if no detections
        if len(detections.xyxy) == 0:
            detections.mask = np.array([])
            return detections
        
        # Segment with SAM
        self.sam_predictor.set_image(rgb_img)
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
                hq_token_only=True,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        
        detections.mask = np.array(result_masks)

        # Annotate and publish segmented image
        # self._publish_segmented_image(cv_img, detections)

        return detections
    
    def _publish_segmented_image(self, cv_img, detections):
        """Annotate the image with masks and bounding boxes and publish it."""
        annotated = cv_img.copy()

        # Draw masks
        mask_annotator = sv.MaskAnnotator()
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)

        # Draw bounding boxes with labels (sv 0.6 BoxAnnotator handles labels)
        labels = [
            f"{self.classes[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Publish
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        self.segmented_img_pub.publish(img_msg)

    def apply_segmentation_to_depth(self, depth_img, rgb_img_msg, detections):
        """Apply segmentation masks to depth image and create 3D point cloud with color"""
        # Early exit if no detections
        if len(detections.mask) == 0:
            return
        
        rospy.loginfo("Humans or mobile robot detected!")
        
        # Combine all masks
        binary_image = np.any(detections.mask, axis=0)
        
        # Convert depth image (Gazebo publishes 32FC1 in meters)
        depth_img_cv = self.bridge.imgmsg_to_cv2(img_msg=depth_img, desired_encoding="passthrough")
        depth_img_m = depth_img_cv.astype(np.float32)
        
        # Convert RGB image for color extraction
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_img_msg, "bgr8")
        
        # Retrieve camera intrinsic parameters
        P_matrix = np.array(self.camera_info.P).reshape((3, 4))
        fx = P_matrix[0, 0]
        fy = P_matrix[1, 1]
        cx = P_matrix[0, 2]
        cy = P_matrix[1, 2]
        
        # Create point cloud from segmentation
        v, u = np.where(binary_image)
        z = depth_img_m[v, u]
        
        # Filter out invalid depths
        valid_mask = z > 0
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]
        
        # Compute 3D coordinates
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        segmented_depth_points = np.stack([x, y, z], axis=-1)
        
        # Extract BGR color for each point
        point_colors = rgb_cv[v, u]  # shape (N, 3) in BGR
        
        # Filter points by distance (max 2.5 meters from camera)
        distances = np.linalg.norm(segmented_depth_points, axis=1)
        distance_mask = distances <= 2.5
        segmented_depth_points = segmented_depth_points[distance_mask]
        point_colors = point_colors[distance_mask]
        
        rospy.loginfo(f"Generated {len(segmented_depth_points)} 3D points from segmentation (within 2.5m)")
        
        # Create and publish PointCloud2 message
        # pointcloud_msg = self.create_pointcloud2(
        #     segmented_depth_points,
        #     header=depth_img.header, 
        #     frame_id=self.camera_info.header.frame_id 
        # )
        # self.pointcloud_pub.publish(pointcloud_msg)
        
        return segmented_depth_points, point_colors, depth_img.header
    
    def create_and_publish_projected_map(self, points_3d, colors, header, target_frame="map"):
        """Create and publish 2D projection of segmented 3D points in global frame"""
        if len(points_3d) == 0:
            return
        
        if self.octomap_projected_map_img is None:
            rospy.logwarn("No static background available")
            return
        
        try:
            # Use the static background from the first octomap projected_map
            projected_img = self.octomap_projected_map_img.copy()
            # projected_img = np.zeros_like(self.octomap_projected_map_img)
            
            # Get map info
            map_width = self.octomap_map_info['width']
            map_height = self.octomap_map_info['height']
            map_frame = self.octomap_map_info['frame_id']
            
            # Get transform from camera frame to map frame
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                header.frame_id,
                header.stamp,
                rospy.Duration(0.5)
            )
            
            # Transform all points to global frame
            points_global = []
            colors_global = []
            for i, point in enumerate(points_3d):
                point_stamped = PointStamped()
                point_stamped.header = header
                point_stamped.point.x = point[0]
                point_stamped.point.y = point[1]
                point_stamped.point.z = point[2]
                
                # Transform to global frame
                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                points_global.append([
                    transformed_point.point.x,
                    transformed_point.point.y,
                    transformed_point.point.z
                ])
                colors_global.append(colors[i])
            
            points_global = np.array(points_global)
            colors_global = np.array(colors_global)
            
            # Get map resolution and origin from stored OccupancyGrid metadata
            map_resolution = self.octomap_map_info['resolution']
            origin_x = self.octomap_map_info['origin_x']
            origin_y = self.octomap_map_info['origin_y']
            
            for i, point in enumerate(points_global):
                # Convert world coordinates to pixel coordinates
                # OccupancyGrid: pixel (0,0) corresponds to world (origin_x, origin_y)
                px = int((point[0] - origin_x) / map_resolution)
                py_raw = int((point[1] - origin_y) / map_resolution)
                # Account for vertical flip applied in occupancy_grid_to_image
                py = map_height - 1 - py_raw
                
                # Check if point is within image bounds
                if 0 <= px < map_width and 0 <= py < map_height:
                    # Draw point in red (BGR: 0, 0, 255)
                    cv2.circle(projected_img, (px, py), 2, (0, 0, 255), -1)
            
            # Convert to ROS Image message and publish
            projected_img_msg = self.bridge.cv2_to_imgmsg(projected_img, encoding="bgr8")
            projected_img_msg.header.stamp = rospy.Time.now()
            projected_img_msg.header.frame_id = target_frame
            
            self.projected_img_pub.publish(projected_img_msg)
            rospy.loginfo(f"Published projected map with {len(points_global)} points")
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF transform failed: {e}")
  
    def camera_info_callback(self, msg):
        """Callback for camera info - cached once received"""
        if self.camera_info is None:
            self.camera_info = msg
            rospy.loginfo("Camera info received and cached")
    
    def synchronized_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth messages"""
        if self.camera_info is None:
            rospy.logwarn("Camera info not yet received, skipping frame")
            return
        
        # Segment the RGB image
        detections = self.segment_image(rgb_msg)
        
        # Apply segmentation to depth and get 3D points with color
        result = self.apply_segmentation_to_depth(depth_msg, rgb_msg, detections)
        
        # Create and publish projected map if points were generated
        if result is not None:
            points_3d, point_colors, points_header = result
            self.create_and_publish_projected_map(points_3d, point_colors, points_header)
    
    def run(self):
        """Start the ROS node spinning"""
        rospy.loginfo("Subscribers ready, starting to process images...")
        rospy.spin()


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node("segmentation_node")
    rospy.loginfo("Node Created")
    
    # Create and run the segmentation node
    node = SegmentationNode()
    node.run()