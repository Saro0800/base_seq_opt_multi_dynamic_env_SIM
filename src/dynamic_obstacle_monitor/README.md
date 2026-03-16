# dynamic_obstacle_monitor

A ROS package for **real-time detection and classification of dynamic obstacles** in a mobile robot's camera frustum. It fuses an OctoMap-based 2-D projected occupancy grid with a segmented colour mask to determine whether a navigation target is obstructed and whether the obstruction is caused by a mobile agent (e.g. a person) or a static obstacle.

## How It Works

1. **`monitor.py`** provides the `DynamicObstacleMonitor` class, which subscribes to the OctoMap projected occupancy grid (`/locobot/octomap_server/projected_map`) and a segmented map image (`/segmented_projected_map`). It converts the occupancy grid into a BGR image and offers an API to:
   - Check whether a given EE target pose is **obstructed**, restricting the check to the camera frustum.
   - Classify the obstacle as **mobile** (red pixels in the segmented mask indicate a detected dynamic agent) or **static** (occupied cells in the OctoMap only).
   - Clear a 15×15 px area around the robot's own footprint to suppress self-detection artifacts.

2. **`octomap_frustum_viewer.py`** is a visualization node (`octomap_frustum_viewer`) that renders a real-time OpenCV window compositing:
   - The initial (static) projected map at reduced opacity as a background layer; this is the map used to ensure reachability during the optimization process.
   - The live projected map shown **only inside the robot's camera frustum** cone.
   - Red overlay pixels from the segmented mask (dynamic obstacles) painted within the frustum, only if mobile agents are detected.
   - A red dot + heading arrow marking the robot's current TF pose.
   - A blue dot + heading arrow marking the current navigation base target (from `/locobot/current_base_target`).


### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| Camera HFOV | 87° | Horizontal field of view (Intel RealSense D435) |
| Frustum range | 2.0 m | Maximum depth range for the frustum check |
| Occupancy threshold | 50 | OccupancyGrid cell value above which a cell is considered occupied |
| Check radius | 0.20 m | Radius around the target within which obstruction is evaluated |
| Segmented mask max age | 20.0 s | Maximum age of the segmented image before it is considered stale |
| Robot clear area | 15×15 px | Square cleared around the robot footprint to avoid self-detection |
