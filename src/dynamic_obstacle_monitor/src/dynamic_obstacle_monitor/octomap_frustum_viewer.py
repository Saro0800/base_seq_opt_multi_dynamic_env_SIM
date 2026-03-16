#!/usr/bin/env python3
"""Octomap frustum viewer with dynamic obstacle overlay.

This module provides a visualization tool that displays the robot's camera
frustum overlaid on the octomap projected map, highlighting dynamic obstacles
detected within the field of view.
"""
import rospy
import numpy as np
import cv2
import tf2_ros
from tf.transformations import euler_from_quaternion

# Import reusable components
from dynamic_obstacle_monitor.monitor import (
    DynamicObstacleMonitor,
    PROJECTED_MAP_TOPIC,
    SEGMENTED_MAP_TOPIC,
)
from geometry_msgs.msg import PoseStamped

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MAP_FRAME = "octomap_frame"
BASE_FRAME = "locobot/base_footprint"

# Intel RealSense D435 horizontal FOV (degrees)
CAMERA_HFOV_DEG = 87.0
# Max range drawn for the frustum cone (metres)
FRUSTUM_RANGE_M = 2.0

# Visualization
WINDOW_NAME = "Octomap 2D + Frustum"
BG_ALPHA = 0.5          # transparency of the full-map layer
FRUSTUM_COLOR = (0, 255, 0)   # BGR – green tint for the frustum region
ROBOT_COLOR = (0, 0, 255)     # BGR – red dot for the robot
TARGET_COLOR = (255, 0, 0)    # BGR – blue arrow for the base target
TARGET_ARROW_LEN_M = 0.3     # length of the target heading arrow (metres)
UPDATE_RATE_HZ = 15


# ──────────────────────────────────────────────────────────────────────
# Main node – visualisation
# ──────────────────────────────────────────────────────────────────────
class OctomapFrustumViewer:
    def __init__(self):
        rospy.init_node("octomap_frustum_viewer", anonymous=False)

        # Parameters (overridable via rosparam / launch)
        map_topic = rospy.get_param("~map_topic", PROJECTED_MAP_TOPIC)
        seg_topic = rospy.get_param("~seg_map_topic", SEGMENTED_MAP_TOPIC)
        self.map_frame = rospy.get_param("~map_frame", MAP_FRAME)
        self.base_frame = rospy.get_param("~base_frame", BASE_FRAME)
        self.hfov_deg = rospy.get_param("~camera_hfov_deg", CAMERA_HFOV_DEG)
        self.frustum_range = rospy.get_param("~frustum_range", FRUSTUM_RANGE_M)
        self.bg_alpha = rospy.get_param("~bg_alpha", BG_ALPHA)
        self.update_rate = rospy.get_param("~update_rate", UPDATE_RATE_HZ)

        self.hfov_rad = np.deg2rad(self.hfov_deg)

        # Reusable monitor – handles subscriptions & obstacle checking
        seg_max_age = rospy.get_param("~seg_max_age", 5.0)
        self.monitor = DynamicObstacleMonitor(
            map_topic=map_topic,
            seg_topic=seg_topic,
            seg_max_age_s=seg_max_age,
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Current base target (updated via subscriber)
        self.target_pose = None  # (x, y, yaw) or None

        # Subscribe to the current base target (the next pose being navigated to)
        rospy.Subscriber(
            "/locobot/current_base_target",
            PoseStamped,
            self._target_cb,
            queue_size=1,
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        rospy.loginfo("[octomap_frustum_viewer] Node ready.")

    # ── callbacks ─────────────────────────────────────────────────────
    def _target_cb(self, msg):
        """Store the latest base target pose (replaces the previous one)."""
        p = msg.pose
        q = p.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.target_pose = (p.position.x, p.position.y, yaw)
        rospy.loginfo("[octomap_frustum_viewer] New base target: "
                      "x=%.2f  y=%.2f  yaw=%.1f°",
                      p.position.x, p.position.y, np.rad2deg(yaw))

    # ── helpers ───────────────────────────────────────────────────────
    def _world_to_pixel(self, wx, wy, origin_x, origin_y, resolution, img_h):
        """Convert world coordinates (metres) to pixel coordinates.

        img_h is needed because the image was flipped vertically.
        """
        px = int((wx - origin_x) / resolution)
        py = img_h - 1 - int((wy - origin_y) / resolution)
        return px, py

    def _build_frustum_polygon(self, robot_x, robot_y, robot_yaw,
                               origin_x, origin_y, resolution, img_h,
                               n_arc_pts=30):
        """Return the frustum polygon (in pixel coords) as an Nx1x2 int32 array.

        The polygon is: robot position → arc of the FOV cone → back to robot.
        """
        half_fov = self.hfov_rad / 2.0
        pts = []

        # Robot centre (tip of the cone)
        px, py = self._world_to_pixel(robot_x, robot_y, origin_x, origin_y,
                                      resolution, img_h)
        pts.append((px, py))

        # Arc from -half_fov to +half_fov relative to robot_yaw
        for i in range(n_arc_pts + 1):
            angle = robot_yaw - half_fov + self.hfov_rad * i / n_arc_pts
            wx = robot_x + self.frustum_range * np.cos(angle)
            wy = robot_y + self.frustum_range * np.sin(angle)
            ppx, ppy = self._world_to_pixel(wx, wy, origin_x, origin_y,
                                            resolution, img_h)
            pts.append((ppx, ppy))

        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    # ── TF lookup ─────────────────────────────────────────────────────
    def _get_robot_pose_2d(self):
        """Return (x, y, yaw) of the robot in the map frame, or None."""
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rospy.Time(0),
                timeout=rospy.Duration(0.1),
            )
            t = tf_stamped.transform.translation
            q = tf_stamped.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return t.x, t.y, yaw
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0,
                "[octomap_frustum_viewer] TF lookup failed: %s", str(e))
            return None

    # ── rendering ─────────────────────────────────────────────────────
    def _render(self):
        """Compose the two-layer image and display it."""
        mon = self.monitor
        if not mon.has_map():
            return

        h, w = mon.static_map_img.shape[:2]
        origin_x = mon.grid_info.origin.position.x
        origin_y = mon.grid_info.origin.position.y
        resolution = mon.grid_info.resolution

        # ------ Layer 1: static (initial) map at reduced alpha ------
        canvas = np.full_like(mon.static_map_img, 255, dtype=np.uint8)
        layer1 = cv2.addWeighted(mon.static_map_img, self.bg_alpha,
                                 canvas, 1.0 - self.bg_alpha, 0)

        # ------ Layer 2: LIVE map shown only inside the frustum ------
        pose = self._get_robot_pose_2d()
        if pose is not None and mon.live_map_img is not None:
            rx, ry, ryaw = pose

            frustum_poly = self._build_frustum_polygon(
                rx, ry, ryaw,
                origin_x, origin_y, resolution, h,
            )

            # Create a mask for the frustum region
            frustum_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(frustum_mask, [frustum_poly], 255)

            # Get the live map with a 15×15 px square around the robot cleared
            cleared_live = mon.get_cleared_live_map((rx, ry, ryaw))

            # Inside the frustum → show the CLEARED live map at full opacity
            layer1[frustum_mask == 255] = cleared_live[frustum_mask == 255]

            # Overlay red pixels from segmented projected map inside the frustum
            if mon.is_seg_mask_fresh():
                seg = mon.segmented_img
                # Ensure the segmented image matches the map dimensions
                if seg.shape[:2] == (h, w):
                    # Detect red pixels: high R, low G, low B (in BGR: B<80, G<80, R>150)
                    red_mask = (
                        (seg[:, :, 2] > 150) &  # R channel high
                        (seg[:, :, 1] < 80)  &  # G channel low
                        (seg[:, :, 0] < 80)     # B channel low
                    )
                    # Combine: pixel must be red AND inside the frustum
                    combined_mask = red_mask & (frustum_mask == 255)
                    # Paint those pixels red on the canvas
                    layer1[combined_mask] = (0, 0, 255)  # BGR red

            # Clear a 15×15 px square around the robot (remove self-detection)
            rpx_c, rpy_c = self._world_to_pixel(rx, ry, origin_x, origin_y,
                                                resolution, h)
            hs = mon.ROBOT_CLEAR_HALF_SIZE
            x1 = max(rpx_c - hs, 0)
            x2 = min(rpx_c + hs + 1, w)
            y1 = max(rpy_c - hs, 0)
            y2 = min(rpy_c + hs + 1, h)
            layer1[y1:y2, x1:x2] = (255, 255, 255)  # white = free

            # Subtle green tinted border around the frustum for clarity
            tint_overlay = layer1.copy()
            cv2.polylines(tint_overlay, [frustum_poly], isClosed=True,
                          color=FRUSTUM_COLOR, thickness=2)
            # Also apply a very light green tint inside the frustum
            # green_overlay = np.zeros_like(layer1)
            # green_overlay[:, :] = FRUSTUM_COLOR
            # blended = cv2.addWeighted(layer1, 0.85, green_overlay, 0.15, 0)
            # layer1[frustum_mask == 255] = blended[frustum_mask == 255]

            # Draw the frustum contour
            cv2.polylines(layer1, [frustum_poly], isClosed=True,
                          color=FRUSTUM_COLOR, thickness=2)

            # Draw robot position
            rpx, rpy = self._world_to_pixel(rx, ry, origin_x, origin_y,
                                            resolution, h)
            cv2.circle(layer1, (rpx, rpy), 5, ROBOT_COLOR, -1)

            # Draw heading line
            head_len = int(0.3 / resolution)  # 30 cm line
            hx = rpx + int(head_len * np.cos(-ryaw))  # minus because image Y is flipped
            hy = rpy - int(head_len * np.sin(ryaw))
            # Compute properly with world_to_pixel
            head_wx = rx + 0.3 * np.cos(ryaw)
            head_wy = ry + 0.3 * np.sin(ryaw)
            hx, hy = self._world_to_pixel(head_wx, head_wy, origin_x, origin_y,
                                          resolution, h)
            cv2.arrowedLine(layer1, (rpx, rpy), (hx, hy),
                            ROBOT_COLOR, 2, tipLength=0.3)

        # ------ Draw base target arrow (if any) ------
        if self.target_pose is not None:
            tx, ty, tyaw = self.target_pose
            tpx, tpy = self._world_to_pixel(tx, ty, origin_x, origin_y,
                                            resolution, h)
            # Tip of the heading arrow
            tip_wx = tx + TARGET_ARROW_LEN_M * np.cos(tyaw)
            tip_wy = ty + TARGET_ARROW_LEN_M * np.sin(tyaw)
            tipx, tipy = self._world_to_pixel(tip_wx, tip_wy,
                                              origin_x, origin_y,
                                              resolution, h)
            # Draw a filled circle at the target position
            cv2.circle(layer1, (tpx, tpy), 5, TARGET_COLOR, -1)
            # Draw the heading arrow
            cv2.arrowedLine(layer1, (tpx, tpy), (tipx, tipy),
                            TARGET_COLOR, 2, tipLength=0.3)

        cv2.imshow(WINDOW_NAME, layer1)
        cv2.waitKey(1)

    # ── main loop ─────────────────────────────────────────────────────
    def run(self):
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            self._render()
            rate.sleep()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        viewer = OctomapFrustumViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
