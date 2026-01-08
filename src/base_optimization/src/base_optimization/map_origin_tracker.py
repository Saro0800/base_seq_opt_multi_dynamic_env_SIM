import rospy
from nav_msgs.msg import OccupancyGrid

def print_origin(msg):
    origin = msg.info.origin
    print(origin)

rospy.init_node("map_origin_tracker")
rospy.Subscriber("/locobot/octomap_server/projected_map", OccupancyGrid, callback=print_origin)

rospy.spin()

