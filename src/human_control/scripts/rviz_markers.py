#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RViz Marker Publisher per Robot e Umano.
Pubblica MarkerArray per visualizzare robot e umano in RViz.

Topics pubblicati:
    /visualization_marker_array - MarkerArray con robot e umano

Requisiti:
    - Gazebo in esecuzione con modelli "robot_obstacle" e "human"

Uso:
    rosrun human_control rviz_markers.py
"""

import sys
import math

try:
    import rospy
    from gazebo_msgs.msg import ModelStates
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
except ImportError:
    print("Errore: ROS 1 non trovato.")
    print("   source /opt/ros/noetic/setup.bash")
    sys.exit(1)


def quaternion_rotate_vector(q, v):
    """Ruota un vettore v usando il quaternione q.
    q = (x, y, z, w), v = (vx, vy, vz)
    Ritorna il vettore ruotato.
    """
    # Quaternion rotation: v' = q * v * q^-1
    # Usando la formula diretta per efficienza
    qx, qy, qz, qw = q
    vx, vy, vz = v
    
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    
    # v' = v + qw * t + cross(q.xyz, t)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    
    return (rx, ry, rz)


def quaternion_multiply(q1, q2):
    """Moltiplica due quaternioni q1 * q2.
    q = (x, y, z, w)
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )


def yaw_to_quaternion(yaw):
    """Converte angolo yaw (radianti) in quaternione."""
    return (0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0))


class RVizMarkerPublisher(object):
    """Pubblica marker per robot e umano su RViz."""

    def __init__(self):
        rospy.init_node('rviz_markers', anonymous=False)

        # Publishers
        self.marker_pub = rospy.Publisher(
            '/visualization_marker_array', MarkerArray, queue_size=10)

        # Subscriber per posizioni da Gazebo
        self.model_sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.model_callback)

        # Posizioni correnti
        self.robot_pose = None
        self.human_pose = None

        # Parametri marker robot (mesh locobot)
        self.robot_mesh = rospy.get_param(
            '~robot_mesh',
            'package://human_control/meshes/locobot_intero_sleep_position_reduced.stl')
        self.robot_scale = rospy.get_param('~robot_scale', 1.0)
        self.robot_color = rospy.get_param('~robot_color', [0.2, 0.2, 0.2, 1.0])  # Grigio scuro

        # Parametri marker umano
        self.human_mesh = rospy.get_param(
            '~human_mesh',
            'package://interbotix_xslocobot_gazebo/models/human_female_1/meshes/female.dae')
        self.human_scale = rospy.get_param('~human_scale', 1.0)
        self.human_color = rospy.get_param('~human_color', [0.8, 0.6, 0.4, 1.0])  # Pelle
        
        # Offset locale della mesh umano (nel frame locale del modello)
        # Questi valori compensano il fatto che l'origine della mesh non e' al centro
        self.human_offset_x = rospy.get_param('~human_offset_x', 0.0)
        self.human_offset_y = rospy.get_param('~human_offset_y', 0.0)
        self.human_offset_z = rospy.get_param('~human_offset_z', 0.0)
        
        # Offset rotazione yaw per la mesh (in gradi)
        # Se la mesh guarda verso -Y quando dovrebbe guardare verso +X, serve +90
        yaw_offset_deg = rospy.get_param('~human_yaw_offset', 90.0)
        self.human_yaw_offset = math.radians(yaw_offset_deg)

        # Frame di riferimento
        self.frame_id = rospy.get_param('~frame_id', 'world')

        # Timer per pubblicazione a 30 Hz
        self.timer = rospy.Timer(rospy.Duration(1.0/30.0), self.publish_markers)

        rospy.loginfo('RViz Marker Publisher avviato!')
        rospy.loginfo('  Frame: {}'.format(self.frame_id))
        rospy.loginfo('  Topic: /visualization_marker_array')

    def model_callback(self, msg):
        """Callback per aggiornamento posizione modelli."""
        try:
            # Robot
            if 'robot_obstacle' in msg.name:
                idx = msg.name.index('robot_obstacle')
                self.robot_pose = msg.pose[idx]

            # Umano
            if 'human' in msg.name:
                idx = msg.name.index('human')
                self.human_pose = msg.pose[idx]

        except Exception as e:
            rospy.logerr('Errore model_callback: {}'.format(e))

    def create_robot_marker(self):
        """Crea marker mesh per il robot."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = self.robot_mesh

        if self.robot_pose:
            marker.pose.position.x = self.robot_pose.position.x
            marker.pose.position.y = self.robot_pose.position.y
            marker.pose.position.z = self.robot_pose.position.z
            marker.pose.orientation = self.robot_pose.orientation
        else:
            marker.pose.orientation.w = 1.0

        # Scala mesh
        marker.scale.x = self.robot_scale
        marker.scale.y = self.robot_scale
        marker.scale.z = self.robot_scale

        # Colore
        marker.color.r = self.robot_color[0]
        marker.color.g = self.robot_color[1]
        marker.color.b = self.robot_color[2]
        marker.color.a = self.robot_color[3]

        marker.lifetime = rospy.Duration(0)  # Persistente

        return marker

    def create_human_marker(self):
        """Crea marker mesh per l'umano."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "human"
        marker.id = 1
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = self.human_mesh

        if self.human_pose:
            # Applica offset locale ruotato con l'orientamento del modello
            q_model = (self.human_pose.orientation.x,
                       self.human_pose.orientation.y,
                       self.human_pose.orientation.z,
                       self.human_pose.orientation.w)
            local_offset = (self.human_offset_x, self.human_offset_y, self.human_offset_z)
            rotated_offset = quaternion_rotate_vector(q_model, local_offset)
            
            marker.pose.position.x = self.human_pose.position.x + rotated_offset[0]
            marker.pose.position.y = self.human_pose.position.y + rotated_offset[1]
            marker.pose.position.z = self.human_pose.position.z + rotated_offset[2]
            
            # Applica offset yaw alla rotazione (compensa mesh non allineata)
            q_yaw_offset = yaw_to_quaternion(self.human_yaw_offset)
            q_final = quaternion_multiply(q_model, q_yaw_offset)
            marker.pose.orientation.x = q_final[0]
            marker.pose.orientation.y = q_final[1]
            marker.pose.orientation.z = q_final[2]
            marker.pose.orientation.w = q_final[3]
        else:
            marker.pose.orientation.w = 1.0

        # Scala mesh
        marker.scale.x = self.human_scale
        marker.scale.y = self.human_scale
        marker.scale.z = self.human_scale

        # Colore (usa embedded colors se disponibili)
        marker.color.r = self.human_color[0]
        marker.color.g = self.human_color[1]
        marker.color.b = self.human_color[2]
        marker.color.a = self.human_color[3]
        marker.mesh_use_embedded_materials = True

        marker.lifetime = rospy.Duration(0)  # Persistente

        return marker

    def publish_markers(self, event=None):
        """Pubblica MarkerArray con robot e umano."""
        marker_array = MarkerArray()

        # Aggiungi marker robot
        robot_marker = self.create_robot_marker()
        marker_array.markers.append(robot_marker)

        # Aggiungi marker umano
        human_marker = self.create_human_marker()
        marker_array.markers.append(human_marker)

        self.marker_pub.publish(marker_array)

    def run(self):
        """Main loop."""
        rospy.spin()


if __name__ == "__main__":
    try:
        node = RVizMarkerPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
