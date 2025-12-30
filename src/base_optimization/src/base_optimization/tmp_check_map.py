import rospy
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid

import tf
import tf.transformations
import tf2_ros
import tf
import tf2_geometry_msgs

def occupancy_grid_callback(msg):
    # Convert OccupancyGrid to OpenCV image
    width = msg.info.width
    height = msg.info.height
    
    # Convert data to numpy array
    grid_data = np.array(msg.data, dtype=np.uint8).reshape((height, width))

    # values in grid_data are probabilities from 0 to 100, with -1 for unknown
    # - cells with value 0 are free
    # - cells with value 100 are occupied
    # - cells with value -1 are unknown

    # convert unknown cells to occupied
    image = np.where(grid_data == -1, 100, grid_data)
    
    # convert to a grayscale image where black (0) = occupied, white (255) = free
    image = np.uint8(255 - image * 2.55)
    
    # Flip the image vertically to correct ROS to OpenCV coordinate system
    # ROS: origin at bottom-left, Y-axis up
    # OpenCV: origin at top-left, Y-axis down
    image = cv2.flip(image, 0)
    
    cv2.imshow('Original Occupancy Grid', image)
    cv2.waitKey(1)

    # Inflate obstacles with radius d (in meters)
    robot_rad = 0.2
    obstacle_dist = 0.05
    d_m = robot_rad + obstacle_dist
    resolution = msg.info.resolution
    
    # Inflation radius in pixels
    d = int(d_m / resolution)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d, 2*d+1))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    # Create binary mask where obstacles are white (occupied cells have low values)
    obstacle_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('Obstacle Mask', obstacle_mask)
    cv2.waitKey(1)

    # Dilate obstacles
    inflated_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
    # Find the inflation layer (difference between inflated and original obstacles)
    inflation_layer = cv2.subtract(inflated_mask, obstacle_mask)
  
    # Convert grayscale image to BGR for color visualization
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Color the inflation layer red (BGR format: red = [0, 0, 255])
    image_color[inflation_layer > 0] = [0, 0, 255]
    # Keep original obstacles black
    image_color[obstacle_mask > 0] = [0, 0, 0]
    
    # Display the image
    cv2.imshow('Inflation Layer', image_color)
    cv2.waitKey(1)

    # create an image with all obstacels inflated
    # (only free space is visible as white)
    image_inflated = image
    image_inflated[inflated_mask > 0] = 0

    cv2.imshow('Inflated Occupancy Grid', image_inflated)
    cv2.waitKey(1)

    # Keep only the connected white space containing the robot's position
    # Retrieve robot position in map frame
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    tf_buffer.can_transform("map", "locobot/base_footprint", rospy.Time(0))
    transform = tf_buffer.lookup_transform("map", "locobot/base_footprint", rospy.Time(0), rospy.Duration(1))
    robot_x = transform.transform.translation.x
    robot_y = transform.transform.translation.y
    
    # Convert robot position from map frame to pixel coordinates
    origin_x = msg.info.origin.position.x
    origin_y = msg.info.origin.position.y
    robot_pixel_x = int((robot_x - origin_x) / resolution)
    robot_pixel_y = int((robot_y - origin_y) / resolution)
    
    # Flip y coordinate because we flipped the image
    robot_pixel_y = height - 1 - robot_pixel_y
        
    # Flood fill from robot position
    free_space = image_inflated.copy()
    cv2.floodFill(free_space, None, (robot_pixel_x, robot_pixel_y), 128)
        
    # Keep only the flooded region (value 128), set everything else to black (0)
    image_inflated_robot_space = np.where(free_space == 128, 255, 0).astype(np.uint8)
    cv2.circle(image_inflated_robot_space, (robot_pixel_x, robot_pixel_y), 3, (190), -1)
    
    cv2.imshow('Robot Free Space', image_inflated_robot_space)
    cv2.waitKey(1)



def main():
    rospy.init_node('occupancy_grid_converter', anonymous=True)
    
    # Subscribe to the topic
    rospy.Subscriber('/locobot/octomap_server/projected_map', OccupancyGrid, occupancy_grid_callback)
    
    rospy.spin()

if __name__ == '__main__':
    main()