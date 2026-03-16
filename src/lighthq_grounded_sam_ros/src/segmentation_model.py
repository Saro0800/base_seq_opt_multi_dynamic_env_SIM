import os
import cv2
import numpy as np
import supervision as sv
from cv_bridge import CvBridge

import torch
import torchvision

import rospy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, CameraInfo

import prova_pkg_grounded_sam.groundingdino as GroundingDINO
import prova_pkg_grounded_sam.segment_anything as SAM
import prova_pkg_grounded_sam.LightHQSAM as LightHQSAM

from prova_pkg_grounded_sam.groundingdino.util.inference import Model
from prova_pkg_grounded_sam.segment_anything import SamPredictor
from prova_pkg_grounded_sam.LightHQSAM.setup_light_hqsam import setup_model


class SegmentationModel:
    """
    A class for performing object detection and segmentation using GroundingDINO and LightHQSAM.
    
    Args:
        classes (list): List of class names to detect (e.g., ["mobile robot", "person"])
        box_threshold (float): Confidence threshold for bounding box detection (default: 0.5)
        text_threshold (float): Text threshold for GroundingDINO (default: 0.5)
        nms_threshold (float): Non-maximum suppression threshold (default: 0.8)
        device (str): Device to run inference on ('cuda' or 'cpu', default: auto-detect)
    """
    
    def __init__(self, classes, box_threshold=0.5, text_threshold=0.5, 
                 nms_threshold=0.8, device=None):
        # Set parameters
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Setup paths
        self._setup_paths()
        
        # Build models
        self._build_models()
        
        rospy.loginfo(f"SegmentationModel initialized with classes: {self.classes}")
        rospy.loginfo(f"Running on device: {self.device}")
    
    def _setup_paths(self):
        """Setup configuration and checkpoint paths."""
        # GroundingDINO config and checkpoint
        grounding_dino_folder = os.path.dirname(os.path.abspath(GroundingDINO.__file__))
        self.grounding_dino_config_path = os.path.join(
            grounding_dino_folder, "config/GroundingDINO_SwinT_OGC.py"
        )
        self.grounding_dino_checkpoint_path = os.path.join(
            grounding_dino_folder, "config/groundingdino_swint_ogc.pth"
        )
        
        # LightHQSAM checkpoint
        hqsam_folder = os.path.dirname(os.path.abspath(LightHQSAM.__file__))
        self.hqsam_checkpoint_path = os.path.join(
            hqsam_folder, "config/sam_hq_vit_tiny.pth"
        )
    
    def _build_models(self):
        """Build and load the GroundingDINO and LightHQSAM models."""
        # Build the GroundingDINO inference model
        self.grounding_dino_model = Model(
            model_config_path=self.grounding_dino_config_path,
            model_checkpoint_path=self.grounding_dino_checkpoint_path
        )
        
        # Build the LightHQSAM predictor
        checkpoint = torch.load(self.hqsam_checkpoint_path)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=self.device)
        self.sam_predictor = SamPredictor(light_hqsam)
    
    def segment_image(self, img_msg):
        """
        Segment objects in an RGB image.
        
        Args:
            img_msg (sensor_msgs.msg.Image): ROS Image message
            
        Returns:
            detections: Detection results with masks
        """
        # Convert the message to a cv image
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
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
        
        # Convert detections to masks
        detections.mask = self._segment(
            image=cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        
        return detections
    
    def _segment(self, image, xyxy):
        """
        Generate segmentation masks from bounding boxes using SAM.
        
        Args:
            image (np.ndarray): Input image
            xyxy (np.ndarray): Bounding boxes in xyxy format
            
        Returns:
            np.ndarray: Array of segmentation masks
        """
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
                hq_token_only=True,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    def get_combined_mask(self, detections, class_indices=None):
        """
        Get a combined binary mask for specified classes.
        
        Args:
            detections: Detection results with masks
            class_indices (list): List of class indices to include (None = all classes)
            
        Returns:
            np.ndarray: Combined binary mask
        """
        if class_indices is None:
            class_indices = range(len(self.classes))
        
        binary_image = None
        for class_idx in class_indices:
            # Gather all masks that belong to the current class
            class_mask_indices = np.where(detections.class_id == class_idx)[0]
            if len(class_mask_indices) == 0:
                continue
            
            # Combine masks for this class into a single binary image
            combined_mask = np.any(detections.mask[class_mask_indices], axis=0)
            
            # Accumulate all class masks together
            if binary_image is None:
                binary_image = combined_mask
            else:
                binary_image = np.logical_or(binary_image, combined_mask)
        
        return binary_image
    
    def apply_segmentation_to_depth(self, camera_info_msg, depth_img, detections):
        """
        Apply segmentation masks to depth image and generate 3D point cloud.
        
        Args:
            camera_info_msg (sensor_msgs.msg.CameraInfo): Camera info message
            depth_img (sensor_msgs.msg.Image): Depth image message
            detections: Detection results with masks
            
        Returns:
            tuple: (pointcloud_msg, segmented_depth_points) or (None, None) if no valid points
        """
        # Get combined binary mask for all classes
        binary_image = self.get_combined_mask(detections)
        
        if binary_image is None:
            return None, None
        
        # Convert the depth image
        depth_img_cv = self.bridge.imgmsg_to_cv2(img_msg=depth_img, desired_encoding="16UC1")
        depth_img_m = depth_img_cv.astype(np.float32) / 1000.0
        
        # Retrieve intrinsic parameters of the stereo camera
        P_matrix = np.array(camera_info_msg.P).reshape((3, 4))
        fx = P_matrix[0, 0]
        fy = P_matrix[1, 1]
        cx = P_matrix[0, 2]
        cy = P_matrix[1, 2]
        
        # Create a pointcloud from the segmentation    
        v, u = np.where(binary_image > 0)
        z = depth_img_m[v, u]
        
        # Filter out invalid depths (zero or negative)
        valid_mask = z > 0
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]
        
        # Compute 3D coordinates
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        segmented_depth_points = np.stack([x, y, z], axis=-1)
        
        rospy.loginfo(f"Generated {len(segmented_depth_points)} 3D points from segmentation")
        
        # Create PointCloud2 message
        pointcloud_msg = self._create_pointcloud2(
            segmented_depth_points,
            header=depth_img.header, 
            frame_id=camera_info_msg.header.frame_id 
        )
        
        return pointcloud_msg, segmented_depth_points
    
    def _create_pointcloud2(self, points, header, frame_id):
        """
        Create a PointCloud2 message from 3D points.
        
        Args:
            points (np.ndarray): Nx3 array of 3D points
            header: ROS header
            frame_id (str): Frame ID for the point cloud
            
        Returns:
            sensor_msgs.msg.PointCloud2: Point cloud message
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        
        # Create PointCloud2 message
        pc2_msg = pc2.create_cloud(header, fields, points)
        
        return pc2_msg
    
    def get_annotated_image(self, img_msg, detections):
        """
        Get an annotated image with bounding boxes and masks.
        
        Args:
            img_msg (sensor_msgs.msg.Image): ROS Image message
            detections: Detection results with masks
            
        Returns:
            np.ndarray: Annotated image
        """
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{confidence:0.2f} {self.classes[class_id]}"
            for confidence, class_id
            in zip(detections.confidence, detections.class_id)
        ]
        annotated_image = mask_annotator.annotate(scene=cv_img.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_image