import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from base_optimization.srv import images_giver

bridge = CvBridge()

def get_latest_images():
    rospy.wait_for_service('images_giver_service')
    try:
        proxy = rospy.ServiceProxy('images_giver_service', images_giver)
        resp = proxy()
        rgb_img = bridge.imgmsg_to_cv2(resp.color_img, 'bgr8')
        depth_img = bridge.imgmsg_to_cv2(resp.depth_img, desired_encoding='passthrough')
        cam_info = resp.cam_info
        return rgb_img, depth_img, cam_info
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return None, None, None

def publish_marker(marker_pub, position, frame_id, marker_id=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "clicked_points"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)

def mouse_callback(event, x, y, flags, param):
    latest_rgb, latest_depth, latest_cam_info = param['latest_data']
    last_clicked_coords = param['last_clicked_coords']
    tf_buffer = param['tf_buffer']
    marker_pub = param['marker_pub']
    target_frame = param['target_frame']

    if event == cv2.EVENT_LBUTTONDOWN and latest_depth is not None and latest_cam_info is not None:
        depth = latest_depth[y, x].astype(np.float32) / 1000.0  # depth in meters
        if depth == 0 or np.isnan(depth):
            print("No valid depth at clicked point.")
            return

        K = np.array(latest_cam_info.K).reshape(3,3)
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        print(f"Clicked pixel: ({x}, {y}) → Camera frame: ({X:.3f}, {Y:.3f}, {Z:.3f}) meters")
        camera_frame = latest_cam_info.header.frame_id

        point_cam = PointStamped()
        point_cam.header.stamp = rospy.Time.now()
        point_cam.header.frame_id = camera_frame
        point_cam.point.x = X
        point_cam.point.y = Y
        point_cam.point.z = Z

        try:
            point_map = tf_buffer.transform(point_cam, target_frame, timeout=rospy.Duration(1.0))
            xyz = [point_map.point.x, point_map.point.y, point_map.point.z]
            print(f"Transformed to '{target_frame}': ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) meters")
            publish_marker(marker_pub, xyz, target_frame)
            last_clicked_coords[0] = xyz  # Save last clicked fixed-frame coordinates
        except Exception as e:
            print(f"TF transform failed: {e}")

def visualize_and_click_service():
    rospy.init_node('locobot_rs_click_to_3d_rviz_service', anonymous=True)
    target_frame = rospy.get_param("~target_frame", "map")

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    marker_pub = rospy.Publisher('clicked_point_marker', Marker, queue_size=10)

    print(f"Click a point in the RGB image window to get its 3D coordinates in the '{target_frame}' frame and visualize in RViz.")

    cv2.namedWindow('RGB Image')
    latest_data = [None, None, None]
    last_clicked_coords = [None]  # Store last clicked fixed-frame coordinates

    cv2.setMouseCallback('RGB Image', mouse_callback, {
        'latest_data': latest_data,
        'last_clicked_coords': last_clicked_coords,
        'tf_buffer': tf_buffer,
        'marker_pub': marker_pub,
        'target_frame': target_frame
    })

    while cv2.getWindowProperty('RGB Image', cv2.WND_PROP_VISIBLE) >= 1 and not rospy.is_shutdown():
        rgb_img, depth_img, cam_info = get_latest_images()
        if rgb_img is not None:
            latest_data[0] = rgb_img
            latest_data[1] = depth_img
            latest_data[2] = cam_info
            cv2.imshow('RGB Image', rgb_img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

    # Return or print the last clicked coordinates in the fixed frame
    if last_clicked_coords[0] is not None:
        print(f"\nLast clicked point in '{target_frame}' frame: {last_clicked_coords[0]}")
        return last_clicked_coords[0]
    else:
        print("\nNo point was clicked.")
        return None

# Usage:
# coords = visualize_and_click_service()
# print("Final returned coordinates:", coords)