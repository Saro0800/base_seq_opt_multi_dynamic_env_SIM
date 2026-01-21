import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image

def visualize_depth(img_msg):
    bridge = CvBridge()

    cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="16UC1")

    print(cv_img[10, 100]/1000)



def visualize_rgb(img_msg):
    bridge = CvBridge()

    cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

    cv2.imshow("RGB image", cv_img)
    cv2.waitKey(0)



if __name__=="__main__":
    # create a ros topic
    rospy.init_node("segment_scene_node")

    # subscribe to the topics to receive the RGB image
    rospy.Subscriber("/locobot/camera/color/image_raw", Image, callback=visualize_rgb)

    # subscribe to the topic to get the depth image
    rospy.Subscriber("/locobot/camera/aligned_depth_to_color/image_raw", Image, callback=visualize_depth)

    rospy.spin()


