#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseArray, Pose
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_gripper_poses(json_file, parent_frame="map"):
    rospy.init_node('gripper_pose_publisher')
    
    # Create the publisher for PoseArray
    pub = rospy.Publisher('gripper_poses', PoseArray, queue_size=10)
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    rospy.sleep(1)  # Wait for the publisher to be ready
    
    while not rospy.is_shutdown():
        # Ask the user for the tag range
        print("\n" + "="*50)
        print("Enter the tag range to load:")
        try:
            tag_start = int(input("  Start tag: "))
            tag_end = int(input("  End tag: "))
        except ValueError:
            print("Error: please enter valid integer numbers!")
            continue
        except KeyboardInterrupt:
            print("\nUser interruption. Exiting...")
            break
        
        if tag_start > tag_end:
            print(f"Error: start tag ({tag_start}) is greater than end tag ({tag_end})!")
            continue
        
        # Create the PoseArray message
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = parent_frame
        
        # Count the tags added
        tags_added = 0
        
        for tag_id, tag_data in data.items():
            # Check if the tag is in the specified range
            if int(tag_id) < tag_start or int(tag_id) > tag_end:
                continue
            
            tag_pos = np.array(tag_data['position'])
            # tag_pos[0] = tag_pos[0] - 0.18
            # tag_pos[2] = tag_pos[2] + 0.03
            tag_quat = tag_data['orientation']  # [x, y, z, w]
            
            # Convert quaternion to rotation matrix
            tag_rot = R.from_quat(tag_quat)
            tag_rot_matrix = tag_rot.as_matrix()
            
            # Relative rotation from tag frame to gripper frame
            R_tag_to_gripper = np.array([
                [ 0, -1,  0],  # X_gripper = -Z_tag, Y_gripper = -X_tag, Z_gripper = +Y_tag
                [ 0,  0,  1],
                [-1,  0,  0]
            ])
            
            # Gripper rotation in the global frame
            gripper_rot_matrix = tag_rot_matrix @ R_tag_to_gripper
            gripper_rot = R.from_matrix(gripper_rot_matrix)
            gripper_quat = gripper_rot.as_quat()  # [x, y, z, w]
            
            # Gripper position: 5cm along +Z of tag_x
            offset_in_tag_frame = np.array([0, 0, 0.05])
            offset_in_global_frame = tag_rot_matrix @ offset_in_tag_frame
            gripper_pos = tag_pos + offset_in_global_frame
            
            # Create Pose for this gripper
            pose = Pose()
            pose.position.x = gripper_pos[0]
            pose.position.y = gripper_pos[1]
            pose.position.z = gripper_pos[2]
            
            pose.orientation.x = gripper_quat[0]
            pose.orientation.y = gripper_quat[1]
            pose.orientation.z = gripper_quat[2]
            pose.orientation.w = gripper_quat[3]
            
            pose_array.poses.append(pose)
            tags_added += 1
            
            rospy.loginfo(f"Added gripper pose for tag_{tag_id}")
        
        if tags_added == 0:
            print(f"No tags found in range [{tag_start}, {tag_end}]")
        else:
            # Publish the PoseArray
            rospy.loginfo(f"Publishing {len(pose_array.poses)} gripper poses on topic 'gripper_poses'")
            pose_array.header.stamp = rospy.Time.now()
            pub.publish(pose_array)
            rospy.sleep(0.5)
            print(f"\n\u2713 Published {tags_added} gripper poses (tags {tag_start}-{tag_end})")
        
        # Ask if continue
        print("\n" + "-"*50)
        try:
            answer = input("Do you want to load more tags? (y/n): ").lower().strip()
            if answer not in ['y', 'yes']:
                print("Terminating program...")
                break
        except KeyboardInterrupt:
            print("\nUser interruption. Exiting...")
            break
    

if __name__ == '__main__':
    try:
        compute_gripper_poses('tag_poses_SIM.json', parent_frame="octomap_frame")
    except rospy.ROSInterruptException:
        pass