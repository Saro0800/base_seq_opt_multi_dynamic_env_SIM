import rospy
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from actionlib_msgs.msg import GoalStatusArray, GoalID
from move_base_msgs.msg import MoveBaseActionGoal

def find_obst_cluster(pose_msg):
    print()
    # wait for a full stop before checking the occupancy grid
    rospy.sleep(1)
    
    rospy.loginfo("Waiting for an occupancy grid msg...")
    grid_msg = rospy.wait_for_message("/locobot/octomap_server/projected_map", OccupancyGrid)
    
    resolution = grid_msg.info.resolution
    height = grid_msg.info.height
    width = grid_msg.info.width
    data = grid_msg.data
    
    # transform to an openCV-ready matrix
    data_np = np.array(data, dtype=np.int8).reshape((height, width))
    data_np = np.where(data_np==-1, 100, data_np)
    grid_img = (255 - data_np*2.55).astype(np.uint8)
    grid_img = cv2.flip(grid_img, 0)
    grid_img_color = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
    
    # convert desired position to pixel coordinates
    pose_x = pose_msg.pose.position.x
    pose_y = pose_msg.pose.position.y
    
    pixel_x = int((pose_x - grid_msg.info.origin.position.x) / resolution)
    pixel_y = height - 1 - int((pose_y - grid_msg.info.origin.position.y) / resolution)
    
    # draw a red circle on the color image and show the entire image
    cv2.circle(grid_img_color, (pixel_x, pixel_y), 3, [0, 0, 255], -1)
    cv2.imshow("", grid_img_color)
    cv2.waitKey(0)
    
    
    

def check_if_remains_valid(move_base_msg):
    pose_msg = move_base_msg.goal.target_pose
    goal_id = move_base_msg.goal_id
    
    rospy.loginfo("Received a new desired base pose")
    rospy.sleep(0.1)
    
    rospy.loginfo("Checking the validity of the desired base pose...")
    
    # get the status of the action
    status_msg = rospy.wait_for_message("/locobot/move_base/status", GoalStatusArray)
    status = status_msg.status_list[-1].status
    
    # while it is active, keep checking
    while status==1:
        
        rospy.sleep(0.1)
    
        # Extract pose position
        pose_x = pose_msg.pose.position.x
        pose_y = pose_msg.pose.position.y
        
        # retrieve the occupancy grid map
        occ_grid_map_msg = rospy.wait_for_message("/locobot/octomap_server/projected_map", OccupancyGrid)
            
        # Get map metadata
        resolution = occ_grid_map_msg.info.resolution
        height = occ_grid_map_msg.info.height
        width = occ_grid_map_msg.info.width
        origin_x = occ_grid_map_msg.info.origin.position.x
        origin_y = occ_grid_map_msg.info.origin.position.y
        
        # get occupancy grid data
        data = occ_grid_map_msg.data
        data_np = np.array(data, dtype=np.int8).reshape((height, width))
        
        # Transform pose to grid coordinates
        grid_x = int((pose_x - origin_x) / resolution)
        grid_y = int((pose_y - origin_y) / resolution)
            
        # check if valid
        if data_np[grid_y, grid_x]>40:
            pose_is_valid = False
            rospy.logwarn("The desired base position is no longer valid")
            
            # stop the movement
            stop_move_topic.publish(goal_id)
            
            find_obst_cluster(pose_msg)
        else:
            pose_is_valid = True
            # rospy.loginfo("The desired base position is still valid")
        
        # get the status of the action
        status_msg = rospy.wait_for_message("/locobot/move_base/status", GoalStatusArray)
        status = status_msg.status_list[-1].status
    
    
    rospy.loginfo("Done checking")
    

rospy.init_node("base_pose_checker_node")
rospy.loginfo("Created base pose validity checker node")

# subscribe to the topic where desired base poses are published
rospy.Subscriber("/locobot/move_base/goal", MoveBaseActionGoal, check_if_remains_valid)
rospy.loginfo("Subscribed to current_goal topic")

# topic to stop the motion of the base
stop_move_topic = rospy.Publisher("/locobot/move_base/cancel", GoalID, queue_size=10)

rospy.loginfo("Waiting for a new mobile base pose...")
rospy.spin()