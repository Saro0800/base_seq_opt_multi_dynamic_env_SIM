# Segmented Projected Map – MQTT Topic Documentation

## Overview

The segmentation client publishes the **projected map with segmented 3D points** on the MQTT topic `segmented_projected_map_mqtt` every time a segmentation frame is processed and projected onto the occupancy grid map.

- **MQTT Broker:** `192.168.0.111:1883` (default, configurable via `--host` / `--port`)
- **Topic:** `segmented_projected_map_mqtt` (configurable via `--topic-seg-map`)
- **QoS:** 0
- **Serialisation:** Python `pickle`
- **Image encoding:** PNG-compressed BGR8

---

## Message Format

Each MQTT message payload is a **`pickle`-serialised Python `dict`** with the following structure:

```python
{
    "type": "segmented_projected_map",   # Message type identifier
    "stamp": float,                       # Unix timestamp (seconds since epoch)
    "image": {
        "data":     bytes,                # PNG-encoded image bytes
        "format":   "png",                # Image compression format
        "height":   int,                  # Image height in pixels
        "width":    int,                  # Image width in pixels
        "channels": 3,                    # Number of channels (always 3)
        "encoding": "bgr8",              # Pixel format: Blue-Green-Red, 8-bit per channel
    },
    "map_info": {
        "width":      int,                # Map width in grid cells (same as image width)
        "height":     int,                # Map height in grid cells (same as image height)
        "resolution": float,              # Metres per pixel/cell
        "origin_x":   float,              # World X coordinate of cell (0,0) in metres
        "origin_y":   float,              # World Y coordinate of cell (0,0) in metres
        "frame_id":   str,                # Coordinate frame (e.g. "map")
    },
    "num_points": int,                    # Number of 3D points projected on this frame
}
```

---

## How to Decode

### Dependencies

```bash
pip install paho-mqtt numpy opencv-python
```

### Minimal Subscriber Example

```python
#!/usr/bin/env python3
"""Subscriber for segmented projected map from MQTT."""

import pickle
import numpy as np
import cv2
import paho.mqtt.client as mqtt

BROKER = "192.168.0.111"   # ← change to your broker address
PORT = 1883
TOPIC = "segmented_projected_map_mqtt"


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker {BROKER}:{PORT}")
        client.subscribe(TOPIC, qos=0)
        print(f"Subscribed to: {TOPIC}")
    else:
        print(f"Connection failed, rc={rc}")


def on_message(client, userdata, mqtt_msg):
    # 1. Deserialise the pickle payload
    msg = pickle.loads(mqtt_msg.payload)

    if msg.get("type") != "segmented_projected_map":
        return

    # 2. Decode the PNG image to a NumPy BGR array
    png_bytes = msg["image"]["data"]
    img_array = np.frombuffer(png_bytes, dtype=np.uint8)
    projected_map = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # projected_map is now a (H, W, 3) BGR uint8 NumPy array

    # 3. Access metadata
    map_info = msg["map_info"]
    stamp = msg["stamp"]
    num_points = msg["num_points"]

    print(f"[{stamp:.3f}] Received map {projected_map.shape[1]}x{projected_map.shape[0]}, "
          f"{num_points} segmented points, "
          f"resolution={map_info['resolution']:.3f} m/px")

    # 4. Use the image (display, further processing, etc.)
    cv2.imshow("Segmented Projected Map", projected_map)
    cv2.waitKey(1)

    # ── Example: convert pixel back to world coordinates ──
    # px, py = 150, 200   # pixel coordinates on the image
    # py_raw = map_info["height"] - 1 - py   # undo vertical flip
    # world_x = px * map_info["resolution"] + map_info["origin_x"]
    # world_y = py_raw * map_info["resolution"] + map_info["origin_y"]


def main():
    client = mqtt.Client(client_id="seg_map_subscriber")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, keepalive=60)
    print(f"Connecting to {BROKER}:{PORT} ...")

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

## Image Content Description

The published image is a **BGR occupancy grid map** with segmented object points drawn on top:

| Element | Color (BGR) | Description |
|---------|-------------|-------------|
| Free space | White `(255, 255, 255)` | Navigable area |
| Occupied | Black `(0, 0, 0)` | Obstacles/walls |
| Unknown | Grey `(127, 127, 127)` | Unexplored area |
| **Segmented points** | **Red `(0, 0, 255)`** | **Detected objects projected from 3D** |

The segmented points (red dots, radius = 2px) represent 3D points from detected objects (people, robots, etc.) that have been:
1. Extracted from depth image using segmentation masks (with 10px border erosion)
2. Transformed from camera frame to map frame via TF
3. Projected onto the 2D occupancy grid

---

## Coordinate Conversions

### Pixel → World (on the received image)

The image has been vertically flipped (`cv2.flip(img, 0)`) so that the Y axis points upward in world coordinates. To convert a pixel `(px, py)` on the image back to world metres:

```python
py_raw = map_info["height"] - 1 - py       # undo vertical flip
world_x = px * map_info["resolution"] + map_info["origin_x"]
world_y = py_raw * map_info["resolution"] + map_info["origin_y"]
```

### World → Pixel

```python
px = int((world_x - map_info["origin_x"]) / map_info["resolution"])
py_raw = int((world_y - map_info["origin_y"]) / map_info["resolution"])
py = map_info["height"] - 1 - py_raw       # apply vertical flip
```

---

## Notes

- The message is only published when at least one segmentation mask is detected and projected. If no objects are detected in a frame, nothing is published.
- The `stamp` field is a Python `time.time()` Unix timestamp (seconds since epoch), not a ROS timestamp.
- The image dimensions match the occupancy grid dimensions received from the `image_bridge/map` topic.
- PNG compression is used to reduce MQTT payload size while preserving lossless quality.
