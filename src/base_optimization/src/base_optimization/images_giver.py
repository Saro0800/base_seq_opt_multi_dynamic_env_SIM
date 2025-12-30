import rospy
from sensor_msgs.msg import Image, CameraInfo
from base_optimization.srv import images_giver, images_giverResponse

QUEUE_SIZE = 100
color_imgs = []
depth_imgs = []
camera_infos = []


def color_image_handler(data):
    if len(color_imgs) < QUEUE_SIZE:
        color_imgs.append(data)
    else:
        color_imgs.pop(0)
        color_imgs.append(data)


def depth_image_handler(data):
    if len(depth_imgs) < QUEUE_SIZE:
        depth_imgs.append(data)
    else:
        depth_imgs.pop(0)
        depth_imgs.append(data)


def camera_info_handler(data):
    if len(camera_infos) < QUEUE_SIZE:
        camera_infos.append(data)
    else:
        camera_infos.pop(0)
        camera_infos.append(data)

def images_giver_handler(req):
    return images_giverResponse(color_imgs.pop(),
                                depth_imgs.pop(),
                                camera_infos.pop())

rospy.init_node("images_giver_node")

# subscribe to the needed topics
color_image = rospy.Subscriber(
    "/locobot/camera/color/image_raw", Image, callback=color_image_handler)
depth_image = rospy.Subscriber(
    "/locobot/camera/aligned_depth_to_color/image_raw", Image, callback=depth_image_handler)
camera_info = rospy.Subscriber(
    "/locobot/camera/color/camera_info", CameraInfo, callback=camera_info_handler)

# start the service
service = rospy.Service("images_giver_service", images_giver, images_giver_handler)
rospy.spin()