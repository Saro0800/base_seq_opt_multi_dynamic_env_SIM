import numpy as np
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray, Point
import tf.transformations

def sample_sphere_fibonacci_grid(center, radius, n_samples=50):
    x, y, z = center[0], center[1], center[2]

    samples = []

    r_phi = np.pi * (np.sqrt(5.) - 1.)

    for i in range(n_samples):
        ys = 1 - (i/float(n_samples - 1)) * 2

        r = np.sqrt(1 - ys**2)
        theta = r_phi * i

        xs = np.cos(theta) * r
        zs = np.sin(theta) * r

        s = [x + radius * xs, y + radius * ys, z + radius * zs]
        samples.append(s)

    return samples


def gen_poses(center, radius, sphere_samples):
    # center coordinates
    xc, yc, zc = center[0], center[1], center[2]
    '''
        RPY sono gli angoli rispetto al sistema di riferimento in cui sono
        stati definiti i centri delle sfere (punti raggiungibili)
    '''
    pose_angles_RPY = []

    for i in range(len(sphere_samples)):
        xs, ys, zs = sphere_samples[i,0], sphere_samples[i, 1], sphere_samples[i, 2]

        # compute the vector parallel to the line going from a sample
        # towards the center of the sphere
        v = [xc-xs, yc-ys, zc-zs]

        # normalize the vector
        v_norm = v / np.linalg.norm(v, ord=2)

        # compute the yaw angle
        yaw = np.arctan2(v_norm[1], v_norm[0])

        # compute the pitch angle
        pitch = -np.arctan2(v_norm[2],
                            np.sqrt(v_norm[1]**2 + v_norm[0]**2))

        # set the roll angle to 0
        roll = 0.

        pose_angles_RPY.append([roll, pitch, yaw])

    return pose_angles_RPY

if __name__=="__main__":
    rospy.init_node("vis_sphere_and_poses")
    pub_center = rospy.Publisher("vis_center", Marker, queue_size=10)
    pub_sphere = rospy.Publisher("vis_sphere", Marker, queue_size=10)
    pub_poses = rospy.Publisher("vis_poses", PoseArray, queue_size=10)
    
    c = np.zeros(3)
    r = 0.3
    sphere_samples = sample_sphere_fibonacci_grid(
        center=c, radius=r, n_samples=20)
    sphere_samples = np.array(sphere_samples)
    rospy.loginfo("Generated sampling of each sphere...")

    # generate a pose for each samples.
    # the orientation of the poses is always the same
    pose_angles_RPY = gen_poses(
        center=c, radius=r, sphere_samples=sphere_samples)
    pose_angles_RPY = np.array(pose_angles_RPY)
    rospy.loginfo("Generated all target poses...")
    
    # create the center Marker message
    center_msg = Marker()
    center_msg.header.frame_id = "map"
    center_msg.header.stamp = rospy.Time.now()
    center_msg.color.r = 0.0
    center_msg.color.g = 0.0
    center_msg.color.b = 1.0
    center_msg.color.a = 1.0
    center_msg.type = Marker.SPHERE
    center_msg.pose.position.x = c[0]
    center_msg.pose.position.y = c[1]
    center_msg.pose.position.z = c[2]
    center_msg.pose.orientation.w = 1.0
    center_msg.scale.x = 0.1
    center_msg.scale.y = 0.1
    center_msg.scale.z = 0.1
    
    # create the sphere Marker message
    sphere_msg = Marker()
    sphere_msg.header.frame_id = "map"
    sphere_msg.header.stamp = rospy.Time.now()
    sphere_msg.color.r = 211/255
    sphere_msg.color.g = 211/255
    sphere_msg.color.b = 211/255
    sphere_msg.color.a = 0.2
    sphere_msg.type = Marker.SPHERE
    sphere_msg.pose.position.x = c[0]
    sphere_msg.pose.position.y = c[1]
    sphere_msg.pose.position.z = c[2]
    sphere_msg.pose.orientation.w = 1.0
    sphere_msg.scale.x = 2*r
    sphere_msg.scale.y = 2*r
    sphere_msg.scale.z = 2*r
    
    # create the PoseArray message
    poseArray_msg = PoseArray()
    poseArray_msg.header.frame_id = "map"
    poseArray_msg.header.stamp = rospy.Time.now()
    
    for i in range(pose_angles_RPY.shape[0]):
        pose = Pose()
        pose.position.x = c[0]
        pose.position.y = c[1]
        pose.position.z = c[2]
        # pose.position.x = sphere_samples[i,0]
        # pose.position.y = sphere_samples[i,1]
        # pose.position.z = sphere_samples[i,2]
        r, p, y = pose_angles_RPY[i]
        quat = tf.transformations.quaternion_from_euler(r,p,y)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        poseArray_msg.poses.append(pose)
    
    while not rospy.is_shutdown():
        pub_center.publish(center_msg)
        rospy.sleep(0.1)
        pub_sphere.publish(sphere_msg)
        rospy.sleep(0.1)
        pub_poses.publish(poseArray_msg)
        rospy.sleep(0.1)
        