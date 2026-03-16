# Image Bridge – Client Deserialization Guide

## Overview

The `image_bridge_node` streams synchronised **depth + color** image pairs
together with **camera intrinsics**, **camera→map TF transform** and
**occupancy grid map** over **MQTT** using Python `pickle`.

- **MQTT broker:** `192.168.0.111:1883` (configurable)
- **Frame topic:** `image_bridge/frame` (~30 Hz, camera rate)
- **Map topic:** `image_bridge/map` (sent on change)

---

## MQTT Topics

Each MQTT message payload is a `pickle`-serialised Python `dict`.

### `image_bridge/frame` – Image pair + metadata (~30 Hz)

```python
{
    "type": "frame",
    "stamp": float,                    # ROS timestamp (seconds)
    "color": {
        "data":     bytes,             # raw pixel buffer
        "height":   int,               # e.g. 480
        "width":    int,               # e.g. 640
        "encoding": str,               # e.g. "rgb8" or "bgr8"
        "step":     int,               # row stride in bytes
    },
    "depth": {
        "data":     bytes,
        "height":   int,
        "width":    int,
        "encoding": str,               # "32FC1" (metres) or "16UC1" (mm)
        "step":     int,
    },
    "camera_info": {                   # None until first CameraInfo arrives
        "P":        [float] * 12,      # 3×4 projection matrix, row-major
        "K":        [float] * 9,       # 3×3 intrinsic matrix, row-major
        "width":    int,
        "height":   int,
        "frame_id": str,
    },
    "tf_cam_to_map": {                 # None if TF lookup fails
        "translation": [x, y, z],      # metres
        "rotation":    [x, y, z, w],   # quaternion
        "parent_frame": str,           # e.g. "map"
        "child_frame":  str,           # e.g. "locobot/camera_color_optical_frame"
    },
}
```

#### Extracting camera intrinsics

```python
import numpy as np

P = np.array(frame["camera_info"]["P"]).reshape(3, 4)
fx, fy = P[0, 0], P[1, 1]
cx, cy = P[0, 2], P[1, 2]
```

#### Using the TF transform (camera→map)

```python
from scipy.spatial.transform import Rotation

tf = frame["tf_cam_to_map"]
R = Rotation.from_quat(tf["rotation"]).as_matrix()  # 3×3
t = np.array(tf["translation"])                       # 3,

# Transform point from camera frame to map frame
point_map = R @ point_camera + t
```

### `image_bridge/map` – OccupancyGrid (sent on change)

```python
{
    "type": "map",
    "stamp": float,
    "map": {
        "data":       bytes,     # int8 occupancy values (row-major)
        "width":      int,       # grid cells
        "height":     int,       # grid cells
        "resolution": float,     # metres per cell
        "origin_x":   float,     # world X of cell (0,0) in metres
        "origin_y":   float,     # world Y of cell (0,0) in metres
        "frame_id":   str,
    },
}
```

#### Decoding the occupancy grid

```python
import numpy as np

m = msg["map"]
grid = np.frombuffer(m["data"], dtype=np.int8).reshape(m["height"], m["width"])
# Values: -1 = unknown, 0 = free, 100 = occupied
```

#### Converting to an image

```python
import cv2, numpy as np

def grid_to_image(grid):
    """int8 grid → BGR uint8 image."""
    img = np.full((*grid.shape, 3), 127, dtype=np.uint8)   # unknown → grey
    known = grid >= 0
    vals = (255 - grid[known].astype(np.float32) * 255.0 / 100.0)
    vals = np.clip(vals, 0, 255).astype(np.uint8)
    img[known, 0] = vals
    img[known, 1] = vals
    img[known, 2] = vals
    img = cv2.flip(img, 0)  # Y-up → image coords
    return img
```

#### Projecting 3D points onto the map image

```python
def world_to_pixel(point_world, map_info, map_height):
    """Convert world (x,y) to pixel (px, py) on the flipped map image."""
    px = int((point_world[0] - map_info["origin_x"]) / map_info["resolution"])
    py_raw = int((point_world[1] - map_info["origin_y"]) / map_info["resolution"])
    py = map_height - 1 - py_raw  # vertical flip
    return px, py
```

---

## Minimal Python Client

```python
#!/usr/bin/env python3
"""Minimal MQTT client that receives and displays frames from image_bridge."""

import pickle

import numpy as np
import cv2
import paho.mqtt.client as mqtt

# ── encoding → numpy dtype mapping ──────────────────────────────────────
ENCODING_TO_DTYPE = {
    "rgb8":  np.uint8,
    "bgr8":  np.uint8,
    "8UC3":  np.uint8,
    "16UC1": np.uint16,
    "32FC1": np.float32,
}

ENCODING_CHANNELS = {
    "rgb8": 3, "bgr8": 3, "8UC3": 3,
    "16UC1": 1, "32FC1": 1,
}

# ── globals for display ─────────────────────────────────────────────────
map_img = None
map_info = None


def decode_image(img_dict: dict) -> np.ndarray:
    """Convert a raw image dict to a numpy array (H×W or H×W×C)."""
    dtype = ENCODING_TO_DTYPE[img_dict["encoding"]]
    channels = ENCODING_CHANNELS[img_dict["encoding"]]
    h, w = img_dict["height"], img_dict["width"]

    arr = np.frombuffer(img_dict["data"], dtype=dtype)
    if channels == 1:
        arr = arr.reshape((h, w))
    else:
        arr = arr.reshape((h, w, channels))
    return arr


def grid_to_image(grid: np.ndarray) -> np.ndarray:
    """int8 occupancy grid → BGR image."""
    img = np.full((*grid.shape, 3), 127, dtype=np.uint8)
    known = grid >= 0
    vals = (255 - grid[known].astype(np.float32) * 255.0 / 100.0)
    vals = np.clip(vals, 0, 255).astype(np.uint8)
    img[known, 0] = vals
    img[known, 1] = vals
    img[known, 2] = vals
    return cv2.flip(img, 0)


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker (rc={rc})")
    client.subscribe("image_bridge/frame", qos=0)
    client.subscribe("image_bridge/map", qos=0)


def on_message(client, userdata, mqtt_msg):
    global map_img, map_info

    msg = pickle.loads(mqtt_msg.payload)

    # ── map message ─────────────────────────────────────────────────
    if msg["type"] == "map":
        m = msg["map"]
        grid = np.frombuffer(m["data"], dtype=np.int8).reshape(
            m["height"], m["width"]
        )
        map_img = grid_to_image(grid)
        map_info = m
        print(f"Map received: {m['width']}x{m['height']}, "
              f"res={m['resolution']:.3f} m/cell")
        cv2.imshow("Map", map_img)
        cv2.waitKey(1)
        return

    # ── frame message ───────────────────────────────────────────────
    assert msg["type"] == "frame"

    # Decode images
    color = decode_image(msg["color"])
    depth = decode_image(msg["depth"])

    # Convert RGB → BGR for OpenCV display (if needed)
    if msg["color"]["encoding"] == "rgb8":
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    # Normalise depth for visualisation
    if msg["depth"]["encoding"] == "16UC1":
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
    else:
        depth_vis = (depth * 255).clip(0, 255).astype(np.uint8)

    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Camera intrinsics (available after first frame)
    if msg["camera_info"] is not None:
        P = np.array(msg["camera_info"]["P"]).reshape(3, 4)
        fx, fy = P[0, 0], P[1, 1]
        cx, cy = P[0, 2], P[1, 2]
        # Use fx, fy, cx, cy for depth → 3D projection …

    # TF transform (camera → map)
    if msg["tf_cam_to_map"] is not None:
        tf = msg["tf_cam_to_map"]
        # tf["translation"] = [x, y, z]
        # tf["rotation"]    = [qx, qy, qz, qw]

    # Display
    cv2.imshow("Color", color)
    cv2.imshow("Depth", depth_color)
    cv2.waitKey(1)


def main():
    BROKER = "192.168.0.111"   # ← change to your broker address
    PORT = 1883

    client = mqtt.Client(client_id="image_bridge_viewer")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, keepalive=60)
    print(f"Connecting to MQTT broker {BROKER}:{PORT} …")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## Dependencies

Install `paho-mqtt` on both publisher and subscriber machines:

```bash
pip install paho-mqtt
```

---

## Quick Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~mqtt_host` | `192.168.0.111` | MQTT broker address |
| `~mqtt_port` | `1883` | MQTT broker port |
| `~mqtt_topic_frame` | `image_bridge/frame` | MQTT topic for image frames |
| `~mqtt_topic_map` | `image_bridge/map` | MQTT topic for occupancy grid |
| `~mqtt_qos` | `0` | MQTT QoS level (0, 1, or 2) |
| `~queue_size` | `5` | Sync subscriber queue |
| `~slop` | `0.1` | Time sync tolerance (s) |
| `~depth_topic` | `/locobot/camera/aligned_depth_to_color/image_raw` | Depth topic |
| `~color_topic` | `/locobot/camera/color/image_raw` | Color topic |
| `~camera_info_topic` | `/locobot/camera/aligned_depth_to_color/camera_info` | CameraInfo topic |
| `~map_topic` | `/locobot/octomap_server/projected_map` | OccupancyGrid topic |
| `~map_frame` | `map` | Target TF frame for camera→world transform |
