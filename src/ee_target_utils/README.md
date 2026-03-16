# ee_target_utils

A ROS package providing utilities for **AprilTag-based end-effector (gripper) target pose generation** on a LoCoBot platform. It detects AprilTags through the robot's hand camera, records their transforms, and computes gripper target poses that are published for downstream motion planning and base-pose optimization.

## How It Works

1. **`apriltag_reader.launch`** starts the `apriltag_ros` continuous detector on the LoCoBot hand camera, detecting up to 50 tag36h11 AprilTags (IDs 0–49, 3 cm size) and broadcasting their TF frames.
2. **`tag_tf_recorder.py`** listens to the TF tree for transforms from `map` to each `tag_<id>` frame (IDs 0–49). It saves all discovered tag positions and orientations to a JSON file (`tag_transforms.json`) every 5 seconds, and performs a final save on shutdown.
3. **`gripper_poses.py`** reads the saved tag poses from the JSON file and, for a user-specified tag range, computes the corresponding gripper target poses by:
   - Applying a fixed rotation (`R_tag_to_gripper`) to transform from the tag frame to the desired gripper orientation.
   - Offsetting the position by 5 cm along the tag's +Z axis.
   - Publishing the resulting `geometry_msgs/PoseArray` on the `/gripper_poses` topic (frame: `octomap_frame`).




