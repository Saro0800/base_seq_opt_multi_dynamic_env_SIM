#!/usr/bin/env python3
"""MQTT-to-ROS bridge for the segmented projected map.

This node subscribes to the MQTT topic ``segmented_projected_map_mqtt``
(where a remote segmentation client publishes the already-built projected
map) and re-publishes it on the ROS topic ``/segmented_projected_map`` so
that downstream ROS nodes (e.g. DynamicObstacleMonitor) can consume it
without any change.

See ``README_segmented_projected_map_mqtt.md`` for the MQTT message format.
"""

import pickle
import threading

import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import paho.mqtt.client as mqtt


# ──────────────────────────────────────────────────────────────────────
# Default configuration (overridable via rosparam)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MQTT_BROKER = "192.168.0.106"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_TOPIC = "segmented_projected_map_mqtt"
DEFAULT_ROS_TOPIC = "/segmented_projected_map"


class SegmentMqttBridgeNode:
    """Bridges the segmented projected map from MQTT to ROS."""

    def __init__(self):
        rospy.init_node("segment_mqtt_bridge_node", anonymous=False)
        rospy.loginfo("Initializing SegmentMqttBridgeNode...")

        # ── Parameters ────────────────────────────────────────────────
        self.mqtt_broker = rospy.get_param("~mqtt_broker", DEFAULT_MQTT_BROKER)
        self.mqtt_port = rospy.get_param("~mqtt_port", DEFAULT_MQTT_PORT)
        self.mqtt_topic = rospy.get_param("~mqtt_topic", DEFAULT_MQTT_TOPIC)
        ros_topic = rospy.get_param("~ros_topic", DEFAULT_ROS_TOPIC)

        # ── ROS publisher ─────────────────────────────────────────────
        self.bridge = CvBridge()
        self.projected_img_pub = rospy.Publisher(
            ros_topic, Image, queue_size=10
        )
        rospy.loginfo("Will publish on ROS topic: %s", ros_topic)

        # ── MQTT client ───────────────────────────────────────────────
        self.mqtt_client = mqtt.Client(client_id="segment_mqtt_bridge_node")
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect
        self.mqtt_client.on_message = self._on_message

        rospy.loginfo(
            "Connecting to MQTT broker %s:%d ...",
            self.mqtt_broker,
            self.mqtt_port,
        )
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)

        # Run the MQTT network loop in a background thread so that it
        # does not block the ROS spin.
        self._mqtt_thread = threading.Thread(
            target=self.mqtt_client.loop_forever, daemon=True
        )
        self._mqtt_thread.start()

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("SegmentMqttBridgeNode ready.")

    # ── MQTT callbacks ────────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            rospy.loginfo(
                "Connected to MQTT broker %s:%d",
                self.mqtt_broker,
                self.mqtt_port,
            )
            client.subscribe(self.mqtt_topic, qos=0)
            rospy.loginfo("Subscribed to MQTT topic: %s", self.mqtt_topic)
        else:
            rospy.logerr("MQTT connection failed (rc=%d)", rc)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            rospy.logwarn(
                "Unexpected MQTT disconnect (rc=%d). Will auto-reconnect.", rc
            )

    def _on_message(self, client, userdata, mqtt_msg):
        """Decode the MQTT payload and republish as a ROS Image."""
        try:
            msg = pickle.loads(mqtt_msg.payload)
        except Exception as e:
            rospy.logwarn("Failed to deserialise MQTT message: %s", e)
            return

        if msg.get("type") != "segmented_projected_map":
            return

        # Decode the PNG image to a BGR NumPy array
        png_bytes = msg["image"]["data"]
        img_array = np.frombuffer(png_bytes, dtype=np.uint8)
        projected_map = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if projected_map is None:
            rospy.logwarn("cv2.imdecode returned None – skipping frame")
            return

        # Build and publish the ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(projected_map, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = msg.get("map_info", {}).get(
            "frame_id", "map"
        )
        self.projected_img_pub.publish(img_msg)

        rospy.loginfo_throttle(
            5.0,
            "Republished segmented projected map (%dx%d, %d points)",
            projected_map.shape[1],
            projected_map.shape[0],
            msg.get("num_points", -1),
        )

    # ── shutdown ──────────────────────────────────────────────────────
    def _shutdown(self):
        rospy.loginfo("Shutting down MQTT client...")
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()

    # ── main loop ─────────────────────────────────────────────────────
    def run(self):
        rospy.spin()


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        node = SegmentMqttBridgeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
