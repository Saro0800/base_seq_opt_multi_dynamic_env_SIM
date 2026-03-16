#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

PROJECTED_MAP_TOPIC = "/locobot/octomap_server/projected_map"
SEGMENTED_MAP_TOPIC = "/segmented_projected_map"


# helper functions
def occupancy_grid_to_image(grid_msg):
    """Convert a nav_msgs/OccupancyGrid to a BGR image (uint8).

    Mapping:
      -1  (unknown)  -> grey  (127, 127, 127)
       0  (free)     -> white (255, 255, 255)
     100  (occupied) -> black (  0,   0,   0)
    """
    w = grid_msg.info.width
    h = grid_msg.info.height
    data = np.array(grid_msg.data, dtype=np.int8).reshape((h, w))

    # default: unknown -> grey
    img = np.full((h, w, 3), 127, dtype=np.uint8)

    # perform the mapping
    vals = 255 - (data[data >= 0].astype(np.float32) * 255.0 / 100.0)
    vals = np.clip(vals, 0, 255).astype(np.uint8)
    img[data >= 0, 0] = vals
    img[data >= 0, 1] = vals
    img[data >= 0, 2] = vals

    # flip vertically (ROS and OpenCV compatibility)
    img = cv2.flip(img, 0)
    return img


def world_to_pixel(wx, wy, origin_x, origin_y, resolution, img_h):
    """Convert world coordinates (metres) to pixel coordinates.

    img_h is needed because the image is flipped vertically.
    """
    px = int((wx - origin_x) / resolution)
    py = img_h - 1 - int((wy - origin_y) / resolution)
    return px, py


class DynamicObstacleMonitor:
    DEFAULT_OCCUPIED_THRESHOLD = 50     # threshold to consider a target as occupied
    DEFAULT_CHECK_RADIUS_M = 0.20       # check circle around target
    DEFAULT_SEG_MAX_AGE_S = 20.0        # ignore older masks
    ROBOT_CLEAR_HALF_SIZE = 7           # half-side of 15×15 px square cleared around robot

    def _init__(self,
                 map_topic=PROJECTED_MAP_TOPIC,
                 seg_topic=SEGMENTED_MAP_TOPIC,
                 occupied_threshold=DEFAULT_OCCUPIED_THRESHOLD,
                 check_radius_m=DEFAULT_CHECK_RADIUS_M,
                 seg_max_age_s=DEFAULT_SEG_MAX_AGE_S):

        self.occupied_threshold = occupied_threshold
        self.check_radius_m = check_radius_m
        self.seg_max_age = rospy.Duration(seg_max_age_s)

        self.live_grid_msg = None               # latest OccupancyGrid
        self.live_map_img = None                # latest projected OccupancyGrid map (BGR image)
        self.grid_info = None                   # OccupancyGrid.info
        self.static_map_img = None              # BGR image of the FIRST projected map
        self.initial_map_received = False

        self.segmented_img = None               # 2D projected segmentation (BGR image)
        self.segmented_img_stamp = None         # time stamp of the latest segmentation mask received

        self.bridge = CvBridge()

        # Subscribers
        self.map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self.map_cb, queue_size=1)
        self.seg_sub = rospy.Subscriber(seg_topic, Image, self.seg_map_cb, queue_size=1)

        rospy.loginfo("[DynamicObstacleMonitor] Subscribed to %s and %s", map_topic, seg_topic)

    # callbacks
    def map_cb(self, msg):
        """Save the latest OccupancyGrid and its BGR image."""
        self.live_grid_msg = msg
        self.grid_info = msg.info
        self.live_map_img = occupancy_grid_to_image(msg)

        if not self.initial_map_received:
            self.static_map_img = self.live_map_img.copy()
            self.initial_map_received = True
            rospy.loginfo("[DynamicObstacleMonitor] Initial map captured (%d x %d, res %.3f m)",
                          msg.info.width, msg.info.height, msg.info.resolution)

    def seg_map_cb(self, msg):
        """Save the latest segmented projected-map image."""
        try:
            self.segmented_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.segmented_img_stamp = msg.header.stamp
        except Exception as e:
            rospy.logwarn("[DynamicObstacleMonitor] Failed to decode segmented map: %s", str(e))

    # public API 
    def has_map(self):
        """Return True once the first projected map has been received."""
        return self.initial_map_received

    def is_seg_mask_fresh(self):
        """Return True if the segmented mask exists and is younger than seg_max_age."""
        return (self.segmented_img is not None
                and self.segmented_img_stamp is not None
                and (rospy.Time.now() - self.segmented_img_stamp) < self.seg_max_age)

    def get_cleared_live_map(self, robot_pose):
        """Return a copy of live_map_img with a 15x15 px square around the robot set to white (free).

        Args:
            robot_pose: tuple (x, y, yaw) - robot position in map frame.

        Returns:
            np.ndarray or None if no map is available.
        """
        if self.live_map_img is None or self.grid_info is None:
            return self.live_map_img

        info = self.grid_info
        h, w = info.height, info.width
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        rx, ry, _ = robot_pose
        rpx, rpy = world_to_pixel(rx, ry, origin_x, origin_y, resolution, h)

        img = self.live_map_img.copy()
        hs = self.ROBOT_CLEAR_HALF_SIZE
        x1 = max(rpx - hs, 0)
        x2 = min(rpx + hs + 1, w)
        y1 = max(rpy - hs, 0)
        y2 = min(rpy + hs + 1, h)
        img[y1:y2, x1:x2] = (255, 255, 255)  # white = free
        return img

    def is_target_obstructed(self, target_pose, robot_pose=None, hfov_rad=None, frustum_range=None):
        """Check if the current target_pose (target base pose) is blocked.

        If robot_pose, hfov_rad, and frustum_range are provided, checks ONLY
        if the target is within the camera frustum. If the target is outside
        the frustum, returns False (not obstructed, because not visible).

        Args:
            target_pose: geometry_msgs/Pose - the navigation target
            robot_pose: tuple (x, y, yaw) - robot position in map frame (optional)
            hfov_rad: float - horizontal field of view in radians (optional)
            frustum_range: float - max frustum range in metres (optional)

        Returns True if the OccupancyGrid has an occupied cell at the target,
        OR if the fresh segmented mask has red pixels there.
        """
        grid = self.live_grid_msg
        if grid is None:
            return False

        info = grid.info
        w, h = info.width, info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        tx = target_pose.position.x
        ty = target_pose.position.y

        # --- Frustum check (if robot pose provided) ---
        if robot_pose is not None and hfov_rad is not None and frustum_range is not None:
            rx, ry, ryaw = robot_pose
            
            # Distance from robot to target
            dx = tx - rx
            dy = ty - ry
            distance = np.sqrt(dx**2 + dy**2)
            
            # If target is beyond frustum range -> not visible
            if distance > frustum_range:
                return False
            
            # Angle from robot heading to target
            angle_to_target = np.arctan2(dy, dx)
            angle_diff = angle_to_target - ryaw
            # Normalize to [-pi, pi]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            # If target is outside horizontal FOV -> not visible
            if abs(angle_diff) > hfov_rad / 2.0:
                return False
            
            # Target is inside frustum -> proceed with obstruction checks

        # Target pixel in image (flipped) coordinates
        tgt_px, tgt_py = world_to_pixel(tx, ty, origin_x, origin_y, resolution, h)

        # Radius of the robot base footprint in pixels
        radius_px = max(1, int(round(self.check_radius_m / resolution)))

        rospy.loginfo("Obstruction check around target pixel (%d,%d) [world (%.2f, %.2f), radius %d px]",
            tgt_px, tgt_py, tx, ty, radius_px)

        # Bounds check (centre must be inside the map)
        if not (0 <= tgt_px < w and 0 <= tgt_py < h):
            rospy.logwarn("Target pixel (%d,%d) out of map bounds", tgt_px, tgt_py)
            return False

        data = np.array(grid.data, dtype=np.int8).reshape((h, w))

        # Robot pixel position (flipped coords) to ignore self-detection
        robot_px, robot_py = None, None
        if robot_pose is not None:
            rx, ry, _ = robot_pose
            robot_px, robot_py = world_to_pixel(rx, ry, origin_x, origin_y, resolution, h)

        hs = self.ROBOT_CLEAR_HALF_SIZE

        # Build circular mask of pixels to check
        y_min = max(tgt_py - radius_px, 0)
        y_max = min(tgt_py + radius_px, h - 1)
        x_min = max(tgt_px - radius_px, 0)
        x_max = min(tgt_px + radius_px, w - 1)

        yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
        circle_mask = (xx - tgt_px) ** 2 + (yy - tgt_py) ** 2 <= radius_px ** 2

        # Exclude pixels inside the robot-clearing square
        if robot_px is not None:
            robot_clear = ((np.abs(xx - robot_px) <= hs) &
                           (np.abs(yy - robot_py) <= hs))
            circle_mask = circle_mask & ~robot_clear

        # If no valid pixels remain (robot is on target), not obstructed
        if not np.any(circle_mask):
            return False

        # --- Check 1: OccupancyGrid occupancy (circle) ---
        grid_rows = h - 1 - yy  # un-flip for raw grid data indexing
        occ_values = data[grid_rows[circle_mask], xx[circle_mask]]
        occupied_count = int(np.sum(occ_values >= self.occupied_threshold))
        if occupied_count > 0:
            rospy.logwarn("OccupancyGrid: %d/%d pixels occupied in circle around target (%d,%d)",
                occupied_count, int(np.sum(circle_mask)), tgt_px, tgt_py)
            return True

        # --- Check 2: segmented mask (red pixels in circle) ---
        # this is needed because sometimes the mask is updated faster
        # than the OccupancyGrid projected map
        if self.is_seg_mask_fresh():
            seg = self.segmented_img
            if seg.shape[:2] == (h, w):
                seg_roi = seg[y_min:y_max + 1, x_min:x_max + 1]
                b_ch = seg_roi[:, :, 0][circle_mask]
                g_ch = seg_roi[:, :, 1][circle_mask]
                r_ch = seg_roi[:, :, 2][circle_mask]
                red_count = int(np.sum((r_ch > 150) & (g_ch < 80) & (b_ch < 80)))
                if red_count > 0:
                    rospy.logwarn(
                        "Segmented mask: %d red pixels in circle around "
                        "target (%d,%d)",
                        red_count, tgt_px, tgt_py)
                    return True
        return False

    def is_obstacle_mobile(self, target_pose):
        """Check if the obstacle at the target position is a mobile agent.

        Returns True if the segmented mask has red pixels at the target
        (indicating a mobile obstacle), False otherwise.

        Args:
            target_pose: geometry_msgs/Pose - the navigation target

        Returns:
            bool: True if mobile agent, False otherwise
        """
        grid = self.live_grid_msg
        if grid is None or not self.is_seg_mask_fresh():
            return False

        info = grid.info
        w, h = info.width, info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        tx = target_pose.position.x
        ty = target_pose.position.y

        # Check segmented mask for red pixels in a circle around the target
        seg = self.segmented_img
        if seg.shape[:2] == (h, w):
            px, py = world_to_pixel(tx, ty, origin_x, origin_y, resolution, h)
            if 0 <= px < w and 0 <= py < h:
                radius_px = max(1, int(round(self.check_radius_m / resolution)))

                y_min = max(py - radius_px, 0)
                y_max = min(py + radius_px, h - 1)
                x_min = max(px - radius_px, 0)
                x_max = min(px + radius_px, w - 1)

                yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
                circle_mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius_px ** 2

                seg_roi = seg[y_min:y_max + 1, x_min:x_max + 1]
                b_ch = seg_roi[:, :, 0][circle_mask]
                g_ch = seg_roi[:, :, 1][circle_mask]
                r_ch = seg_roi[:, :, 2][circle_mask]
                red_count = int(np.sum((r_ch > 150) & (g_ch < 80) & (b_ch < 80)))
                if red_count > 0:
                    return True  # Red pixels found -> mobile agent
        return False  # No red pixels -> static obstacle
