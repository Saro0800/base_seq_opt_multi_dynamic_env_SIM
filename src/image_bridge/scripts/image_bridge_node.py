#!/usr/bin/env python3
"""
image_bridge_node.py

Subscribes to depth/color images, CameraInfo, OccupancyGrid and TF,
synchronises the image pair, and publishes everything over MQTT
using pickle serialization.

Two MQTT topics are used:

  ── image_bridge/map  (published on first receive and whenever the map changes) ──
  {
      "type"  : "map",
      "stamp" : float,
      "map"   : {
          "data"       : bytes,        # int8 occupancy values
          "width"      : int,
          "height"     : int,
          "resolution" : float,        # m/cell
          "origin_x"   : float,        # metres
          "origin_y"   : float,        # metres
          "frame_id"   : str,
      },
  }

  ── image_bridge/frame  (published for each synchronised depth/colour pair) ──
  {
      "type"  : "frame",
      "stamp" : float,
      "color" : { "data": bytes, "height": int, "width": int,
                  "encoding": str, "step": int },
      "depth" : { "data": bytes, "height": int, "width": int,
                  "encoding": str, "step": int },
      "camera_info" : {               # None until received
          "P"       : list[float],     # 12 elements, row-major 3×4
          "K"       : list[float],     # 9 elements, row-major 3×3
          "width"   : int,
          "height"  : int,
          "frame_id": str,
      },
      "tf_cam_to_map" : {             # None if lookup fails
          "translation" : [x, y, z],   # metres
          "rotation"    : [x, y, z, w],# quaternion
          "parent_frame": str,
          "child_frame" : str,
      },
  }
"""

import os
import pickle
import threading

import rospy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
import message_filters
import numpy as np

import paho.mqtt.client as mqtt

# ---- paho-mqtt v2 compatibility ----
try:
    from paho.mqtt.enums import CallbackAPIVersion
    _PAHO_V2 = True
except ImportError:
    _PAHO_V2 = False


class ImageBridgeNode:
    """MQTT publisher that streams synchronised depth + colour frames
    together with camera intrinsics, TF and occupancy grid."""

    def __init__(self):
        rospy.init_node("image_bridge_node", anonymous=False)

        # ---- parameters ----
        self.mqtt_host = rospy.get_param("~mqtt_host", "192.168.0.111")
        self.mqtt_port = rospy.get_param("~mqtt_port", 1883)
        self.mqtt_topic_frame = rospy.get_param("~mqtt_topic_frame", "image_bridge/frame")
        self.mqtt_topic_map = rospy.get_param("~mqtt_topic_map", "image_bridge/map")
        self.mqtt_qos = rospy.get_param("~mqtt_qos", 0)
        self.queue_size = rospy.get_param("~queue_size", 5)
        self.slop = rospy.get_param("~slop", 0.1)
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)  # Hz (max FPS)
        self._min_publish_interval = 1.0 / self.publish_rate
        self._last_frame_publish_time = 0.0

        depth_topic = rospy.get_param(
            "~depth_topic",
            "/locobot/camera/aligned_depth_to_color/image_raw",
        )
        color_topic = rospy.get_param(
            "~color_topic",
            "/locobot/camera/color/image_raw",
        )
        camera_info_topic = rospy.get_param(
            "~camera_info_topic",
            "/locobot/camera/aligned_depth_to_color/camera_info",
        )
        map_topic = rospy.get_param(
            "~map_topic",
            "/locobot/octomap_server/projected_map",
        )
        self.map_frame = rospy.get_param("~map_frame", "map")

        # ---- cached state ----
        self.camera_info_dict = None        # serialisable dict
        self.camera_frame_id = None         # e.g. "locobot/camera_color_optical_frame"
        self._last_map_stamp = None         # used to detect map changes
        self._cached_map_bytes = None       # last pickled map payload (re-sent on reconnect)

        # ---- TF2 ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---- thread lock for MQTT publishes ----
        self._mqtt_lock = threading.Lock()

        # ---- MQTT client ----
        # Unique client-id avoids "client takeover" if two nodes run
        _client_id = "image_bridge_node_{}".format(os.getpid())

        if _PAHO_V2:
            self.mqtt_client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=_client_id,
                protocol=mqtt.MQTTv311,
            )
        else:
            self.mqtt_client = mqtt.Client(
                client_id=_client_id,
                protocol=mqtt.MQTTv311,
            )

        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.max_inflight_messages_set(20)
        self.mqtt_client.max_queued_messages_set(1)          # queue size = 1
        self.mqtt_client.reconnect_delay_set(1, 30)          # 1 s → 30 s exponential backoff
        self.mqtt_connected = False

        rospy.loginfo(
            "ImageBridge: connecting to MQTT broker %s:%d (client_id=%s) …",
            self.mqtt_host,
            self.mqtt_port,
            _client_id,
        )
        self.mqtt_client.connect_async(self.mqtt_host, self.mqtt_port, keepalive=60)
        self.mqtt_client.loop_start()

        # ---- ROS subscribers ----
        # CameraInfo (latched-style: cache once)
        rospy.Subscriber(
            camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1
        )

        # OccupancyGrid (send to clients on receive)
        rospy.Subscriber(
            map_topic, OccupancyGrid, self._map_cb, queue_size=1
        )

        # Synchronised depth + colour (queue_size=1 for real-time, always latest)
        sub_depth = message_filters.Subscriber(depth_topic, Image, queue_size=1, buff_size=2**24)
        sub_color = message_filters.Subscriber(color_topic, Image, queue_size=1, buff_size=2**24)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_depth, sub_color],
            queue_size=1,  # only keep latest pair
            slop=self.slop,
        )
        self.sync.registerCallback(self._image_cb)

        rospy.loginfo(
            "ImageBridge subscribed to:\n"
            "  depth : %s\n"
            "  color : %s\n"
            "  info  : %s\n"
            "  map   : %s",
            depth_topic,
            color_topic,
            camera_info_topic,
            map_topic,
        )

    # ================================================================== #
    # MQTT helpers
    # ================================================================== #
    if _PAHO_V2:
        def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
            if not reason_code.is_failure:
                self.mqtt_connected = True
                rospy.loginfo("ImageBridge: connected to MQTT broker")
                self._resend_cached_map()
            else:
                rospy.logerr("ImageBridge: MQTT connection failed (rc=%s)", reason_code)

        def _on_mqtt_disconnect(self, client, userdata, flags, reason_code, properties):
            self.mqtt_connected = False
            if reason_code.is_failure:
                rospy.logwarn("ImageBridge: unexpected MQTT disconnect (rc=%s), reconnecting…", reason_code)
    else:
        def _on_mqtt_connect(self, client, userdata, flags, rc):
            if rc == 0:
                self.mqtt_connected = True
                rospy.loginfo("ImageBridge: connected to MQTT broker")
                self._resend_cached_map()
            else:
                rospy.logerr("ImageBridge: MQTT connection failed (rc=%d)", rc)

        def _on_mqtt_disconnect(self, client, userdata, rc):
            self.mqtt_connected = False
            if rc != 0:
                rospy.logwarn("ImageBridge: unexpected MQTT disconnect (rc=%d), reconnecting…", rc)

    def _resend_cached_map(self):
        """Re-publish the last map payload after (re)connection."""
        if self._cached_map_bytes is not None:
            self._do_publish(self.mqtt_topic_map, self._cached_map_bytes)
            rospy.loginfo("ImageBridge: cached map re-sent after reconnect")

    def _do_publish(self, topic: str, data: bytes) -> bool:
        """Thread-safe MQTT publish. Returns True on success."""
        with self._mqtt_lock:
            if not self.mqtt_connected:
                return False
            info = self.mqtt_client.publish(topic, data, qos=self.mqtt_qos)
            return info.rc == mqtt.MQTT_ERR_SUCCESS

    def _publish_mqtt(self, topic: str, data: bytes) -> bool:
        """Publish pickled *data* to the given MQTT topic."""
        return self._do_publish(topic, data)

    # ================================================================== #
    # CameraInfo callback
    # ================================================================== #
    def _camera_info_cb(self, msg: CameraInfo):
        """Cache camera intrinsics (received once)."""
        if self.camera_info_dict is not None:
            return  # already cached
        self.camera_frame_id = msg.header.frame_id
        self.camera_info_dict = {
            "P": list(msg.P),          # 12 floats
            "K": list(msg.K),          # 9 floats
            "width": msg.width,
            "height": msg.height,
            "frame_id": msg.header.frame_id,
        }
        rospy.loginfo(
            "ImageBridge: camera info cached (frame=%s, %dx%d)",
            msg.header.frame_id,
            msg.width,
            msg.height,
        )

    # ================================================================== #
    # OccupancyGrid callback
    # ================================================================== #
    def _map_cb(self, msg: OccupancyGrid):
        """Broadcast the occupancy grid whenever it is received / changes."""
        stamp = msg.header.stamp.to_sec()
        if self._last_map_stamp == stamp:
            return  # duplicate

        self._last_map_stamp = stamp

        map_payload = {
            "type": "map",
            "stamp": stamp,
            "map": {
                "data": bytes(np.array(msg.data, dtype=np.int8).tobytes()),
                "width": msg.info.width,
                "height": msg.info.height,
                "resolution": msg.info.resolution,
                "origin_x": msg.info.origin.position.x,
                "origin_y": msg.info.origin.position.y,
                "frame_id": msg.header.frame_id,
            },
        }

        data = pickle.dumps(map_payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._cached_map_bytes = data            # cache for reconnect
        if self._publish_mqtt(self.mqtt_topic_map, data):
            rospy.loginfo(
                "ImageBridge: map sent (%dx%d, res=%.3f)",
                msg.info.width,
                msg.info.height,
                msg.info.resolution,
            )
        else:
            rospy.loginfo(
                "ImageBridge: map cached (%dx%d, res=%.3f), will send on connect",
                msg.info.width,
                msg.info.height,
                msg.info.resolution,
            )

    # ================================================================== #
    # TF lookup helper
    # ================================================================== #
    def _lookup_tf(self, stamp):
        """Return camera→map transform dict, or None on failure."""
        if self.camera_frame_id is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame_id,
                stamp,
                rospy.Duration(0.1),
            )
            tr = t.transform.translation
            ro = t.transform.rotation
            return {
                "translation": [tr.x, tr.y, tr.z],
                "rotation": [ro.x, ro.y, ro.z, ro.w],
                "parent_frame": t.header.frame_id,
                "child_frame": t.child_frame_id,
            }
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(5.0, "ImageBridge TF lookup failed: %s", exc)
            return None

    # ================================================================== #
    # Synchronised image callback
    # ================================================================== #
    def _image_cb(self, depth_msg: Image, color_msg: Image):
        if not self.mqtt_connected:
            return

        # Rate limiting
        now = rospy.get_time()
        if (now - self._last_frame_publish_time) < self._min_publish_interval:
            return
        self._last_frame_publish_time = now

        stamp = depth_msg.header.stamp

        payload = {
            "type": "frame",
            "stamp": stamp.to_sec(),
            "depth": {
                "data": bytes(depth_msg.data),
                "height": depth_msg.height,
                "width": depth_msg.width,
                "encoding": depth_msg.encoding,
                "step": depth_msg.step,
            },
            "color": {
                "data": bytes(color_msg.data),
                "height": color_msg.height,
                "width": color_msg.width,
                "encoding": color_msg.encoding,
                "step": color_msg.step,
            },
            "camera_info": self.camera_info_dict,      # None until first CameraInfo
            "tf_cam_to_map": self._lookup_tf(stamp),   # None on failure
        }

        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._publish_mqtt(self.mqtt_topic_frame, data)

    # ================================================================== #
    # Shutdown
    # ================================================================== #
    def shutdown(self):
        rospy.loginfo("ImageBridge: shutting down MQTT client …")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


def main():
    node = ImageBridgeNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
