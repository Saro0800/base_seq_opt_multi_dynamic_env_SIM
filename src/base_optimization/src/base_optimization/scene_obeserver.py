import rospy
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid

def process_occupancy_grid(grid_msg, visualize=False):
    # Convert the occupancy grid data to a numpy array
    width = grid_msg.info.width
    height = grid_msg.info.height
    data = np.array(grid_msg.data).reshape((height, width))

    # create a mask of free space
    free_space_mask = np.zeros_like(data, dtype=np.uint8)
    free_space_mask = np.where(data == 0, 255, 0).astype(np.uint8)
    free_space_mask = cv2.flip(free_space_mask, 0)

    # create a mask of the occupied spac
    occ_space_mask = np.zeros_like(data, dtype=np.uint8)
    occ_space_mask = np.where(data == 100, 255, 0).astype(np.uint8)
    occ_space_mask = cv2.flip(occ_space_mask, 0)

    # create a mask of unknown space
    unk_space_mask = np.zeros_like(data, dtype=np.uint8)
    unk_space_mask = np.where(data == -1, 255, 0).astype(np.uint8)
    unk_space_mask = cv2.flip(unk_space_mask, 0)

    if visualize:
        cv2.imshow("Free Space Mask", free_space_mask)
        cv2.imshow("Occupied Space Mask", occ_space_mask)
        cv2.imshow("Unknown Space Mask", unk_space_mask)
        cv2.waitKey(3000)

    return free_space_mask, occ_space_mask, unk_space_mask

def resize_and_align_grid(grid_map_data, old_info, new_info):
    # Create a new grid map filled with unknown values (-1)
    new_width = new_info.width
    new_height = new_info.height
    
    old_width = old_info.width
    old_height = old_info.height
    
    resolution = old_info.resolution
    
    new_grid_data = -1 * np.ones((new_height, new_width), dtype=np.int8)

    # compute the offset in pixels between the origins (preserve direction)
    w_diff = new_width - old_width
    h_diff = new_height - old_height
    
    if new_info.origin.position.x < old_info.origin.position.x and \
        new_info.origin.position.y < old_info.origin.position.y:
            rig_start = 0
            col_start = w_diff
    elif new_info.origin.position.x >= old_info.origin.position.x and \
        new_info.origin.position.y < old_info.origin.position.y:
            rig_start = 0
            col_start = 0
    elif new_info.origin.position.x < old_info.origin.position.x and \
        new_info.origin.position.y >= old_info.origin.position.y:
            rig_start = h_diff
            col_start = w_diff
    else:
            rig_start = h_diff
            col_start = 0
    
    # copy the old grid data into the new grid data at the computed offset
    new_grid_data[rig_start:rig_start + old_height, col_start:col_start + old_width] = grid_map_data
    
    return new_grid_data

def detect_new_obstacles(occ_space_mask, min_area=3):
    global init_occ_space_mask
    
    # check new obstacles in the occupied space (use cv2.subtract to avoid underflow)
    new_obst_mask = cv2.subtract(occ_space_mask, init_occ_space_mask)
    
    # create a new mask with filtered obstacles
    filtered_obst_mask = np.zeros_like(new_obst_mask, dtype=np.uint8)
    
    # find counters of new obstacles
    contours, _ = cv2.findContours(new_obst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cntour in contours:
        area = cv2.contourArea(cntour)
        if area > min_area:
            cv2.drawContours(filtered_obst_mask, [cntour], -1, 255, cv2.FILLED)
    
    cv2.imshow("Filtered New Obstacles Mask", filtered_obst_mask)
    cv2.waitKey(1)            
            

def observe_grid_map(grid_msg):
    global init_free_space_mask, init_occ_space_mask, init_unk_space_mask

    # check if the new grid has a different size of the init one
    if grid_msg.info.width != init_grid_msg.info.width or \
       grid_msg.info.height != init_grid_msg.info.height:
        
        old_data_np = np.array(init_grid_msg.data).reshape(
            (init_grid_msg.info.height, init_grid_msg.info.width))
        upd_data = resize_and_align_grid(old_data_np, init_grid_msg.info, grid_msg.info)

        init_grid_msg.info = grid_msg.info
        init_grid_msg.data = upd_data
        
        # process new occupancy grid map
        init_free_space_mask, init_occ_space_mask, init_unk_space_mask = process_occupancy_grid(init_grid_msg)

    # process the new occupancy grid map
    free_space_mask, occ_space_mask, unk_space_mask = process_occupancy_grid(grid_msg)
    
    # detect and filter new obstacles
    detect_new_obstacles(occ_space_mask)
    
    
    

    




# initialize the ROS node
rospy.init_node("scene_observer_node")
rospy.loginfo("Scene Observer Node has started.")

# acquire the first copy of the occupancy grip map
init_grid_msg = rospy.wait_for_message("/locobot/octomap_server/projected_map", OccupancyGrid)
rospy.loginfo("Acquired the first occupancy grid map.")

# process first occupancy grid map
init_free_space_mask, init_occ_space_mask, init_unk_space_mask = process_occupancy_grid(init_grid_msg)
rospy.loginfo("Processed the first occupancy grid map.")

# subscribe to the occupancy grid topic
rospy.Subscriber("/locobot/octomap_server/projected_map", OccupancyGrid, callback=observe_grid_map)

rospy.spin()
