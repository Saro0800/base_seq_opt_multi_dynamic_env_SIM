#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualizzatore Mappa 2D
Mostra la posizione dell'umano su una mappa vista dall'alto.

Requisiti:
  pip3 install opencv-python

Uso:
  rosrun human_control map_viewer.py
"""

import sys
import os
import math

try:
    import cv2
    import numpy as np
except ImportError:
    print("Errore: Installa OpenCV: pip3 install opencv-python")
    sys.exit(1)

try:
    import rospy
    from gazebo_msgs.msg import ModelStates
except ImportError:
    print("Errore: ROS 1 non trovato.")
    print("   source /opt/ros/noetic/setup.bash")
    sys.exit(1)


class MapViewer(object):
    """Visualizzatore mappa 2D con posizione umano."""

    def __init__(self):
        rospy.init_node('map_viewer', anonymous=False)

        # Configurazione mappa di default
        self.map_size_meters = 20.0
        self.map_center_x = 0.0
        self.map_center_y = 0.0
        self.window_size = 600

        # Info mappa da file (se disponibile)
        self.map_info = self.load_map_info()

        # Scala: pixel per metro
        self.scale = self.window_size / self.map_size_meters

        # Posizione corrente umano
        self.human_x = 0.0
        self.human_y = 0.0
        self.human_yaw = 0.0
        self.human_found = False

        # Posizione robot obstacle
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_found = False

        # Traccia percorso
        self.trail = []
        self.robot_trail = []
        self.max_trail_length = 500

        # Carica immagine mappa se esiste
        self.map_image = self.load_map_image()

        # Subscriber per la posizione dei modelli
        self.model_sub = rospy.Subscriber(
            '/gazebo/model_states',
            ModelStates,
            self.model_callback,
            queue_size=1
        )

        rospy.loginfo('Map Viewer avviato!')
        rospy.loginfo('Topic: /gazebo/model_states')
        rospy.loginfo('Premi ESC o Q per uscire, C per cancellare traccia, +/- per zoom')

        # Crea finestra
        cv2.namedWindow('Mappa 2D', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mappa 2D', self.window_size, self.window_size)

    def load_map_info(self):
        """Carica info mappa dal file generato."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        info_path = os.path.join(script_dir, 'map_overhead_info.txt')

        info = {
            'center_x': 0.0,
            'center_y': 0.0,
            'size_meters': 20.0
        }

        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=')
                            try:
                                info[key] = float(value)
                            except ValueError:
                                pass

                self.map_size_meters = info['size_meters']
                self.map_center_x = info['center_x']
                self.map_center_y = info['center_y']

                rospy.loginfo("Info mappa caricate: centro=({:.1f}, {:.1f}), size={:.1f}m".format(
                    self.map_center_x, self.map_center_y, self.map_size_meters))

            except Exception as e:
                rospy.logwarn("Errore lettura info mappa: {}".format(e))

        return info

    def load_map_image(self):
        """Carica immagine mappa se disponibile."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = os.path.join(script_dir, 'map_overhead.png')

        if os.path.exists(map_path):
            rospy.loginfo('Caricata mappa: {}'.format(map_path))
            img = cv2.imread(map_path)
            if img is not None:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return cv2.resize(img, (self.window_size, self.window_size))

        map_path_jpg = os.path.join(script_dir, 'map_overhead.jpg')
        if os.path.exists(map_path_jpg):
            rospy.loginfo('Caricata mappa: {}'.format(map_path_jpg))
            img = cv2.imread(map_path_jpg)
            if img is not None:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return cv2.resize(img, (self.window_size, self.window_size))

        rospy.loginfo('Nessuna immagine mappa trovata, uso griglia')
        return None

    def quaternion_to_yaw(self, q):
        """Converte quaternione in yaw (rotazione attorno a Z)."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def model_callback(self, msg):
        """Callback per aggiornamento posizione modelli."""
        try:
            if 'human' in msg.name:
                idx = msg.name.index('human')
                pose = msg.pose[idx]

                self.human_x = pose.position.x
                self.human_y = pose.position.y
                self.human_yaw = self.quaternion_to_yaw(pose.orientation)
                self.human_found = True

                self.trail.append((self.human_x, self.human_y))
                if len(self.trail) > self.max_trail_length:
                    self.trail.pop(0)

            if 'robot_obstacle' in msg.name:
                idx = msg.name.index('robot_obstacle')
                pose = msg.pose[idx]
                self.robot_x = pose.position.x
                self.robot_y = pose.position.y
                self.robot_yaw = self.quaternion_to_yaw(pose.orientation)
                self.robot_found = True
                self.robot_trail.append((self.robot_x, self.robot_y))
                if len(self.robot_trail) > self.max_trail_length:
                    self.robot_trail.pop(0)

        except Exception as e:
            rospy.logerr('Errore: {}'.format(e))

    def world_to_pixel(self, wx, wy):
        """Converte coordinate mondo Gazebo in coordinate pixel immagine.
        L'immagine mappa è ruotata di 90° antiorario, quindi applichiamo
        la stessa rotazione alle coordinate: (dx, dy) -> (-dy, dx)."""
        cx = self.window_size / 2
        cy = self.window_size / 2

        dx = wx - self.map_center_x
        dy = wy - self.map_center_y

        # Rotazione 90° antioraria: pixel_x = -dy, pixel_y = -dx
        px = int(cx - dy * self.scale)
        py = int(cy - dx * self.scale)

        return px, py

    def draw_grid(self, frame):
        """Disegna griglia di riferimento."""
        grid_color = (50, 50, 50)

        if self.map_size_meters <= 20:
            grid_interval = 1.0
        elif self.map_size_meters <= 50:
            grid_interval = 5.0
        else:
            grid_interval = 10.0

        x_start = math.floor((self.map_center_x - self.map_size_meters/2) / grid_interval) * grid_interval
        x_end = math.ceil((self.map_center_x + self.map_size_meters/2) / grid_interval) * grid_interval

        x = x_start
        while x <= x_end:
            px, _ = self.world_to_pixel(x, 0)
            if 0 <= px < self.window_size:
                color = (80, 80, 80) if x == 0 else grid_color
                thickness = 2 if x == 0 else 1
                cv2.line(frame, (px, 0), (px, self.window_size), color, thickness)
            x += grid_interval

        y_start = math.floor((self.map_center_y - self.map_size_meters/2) / grid_interval) * grid_interval
        y_end = math.ceil((self.map_center_y + self.map_size_meters/2) / grid_interval) * grid_interval

        y = y_start
        while y <= y_end:
            _, py = self.world_to_pixel(0, y)
            if 0 <= py < self.window_size:
                color = (80, 80, 80) if y == 0 else grid_color
                thickness = 2 if y == 0 else 1
                cv2.line(frame, (0, py), (self.window_size, py), color, thickness)
            y += grid_interval

        # Disegna assi X e Y con frecce e label
        origin_px, origin_py = self.world_to_pixel(0, 0)
        axis_len_px = int(2.0 * self.scale)  # 2 metri in pixel
        axis_len_px = max(40, min(axis_len_px, 150))

        # Asse X (rosso) - punta a destra
        if 0 <= origin_py < self.window_size:
            x_end = min(origin_px + axis_len_px, self.window_size - 5)
            if origin_px < self.window_size and x_end > origin_px + 10:
                cv2.arrowedLine(frame, (origin_px, origin_py), (x_end, origin_py),
                               (0, 0, 255), 2, tipLength=0.15)
                cv2.putText(frame, "X", (x_end + 5, origin_py + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Asse Y (verde) - punta in alto (Y+ nel mondo = su nell'immagine)
        if 0 <= origin_px < self.window_size:
            y_end = max(origin_py - axis_len_px, 5)
            if origin_py > 0 and origin_py - y_end > 10:
                cv2.arrowedLine(frame, (origin_px, origin_py), (origin_px, y_end),
                               (0, 255, 0), 2, tipLength=0.15)
                cv2.putText(frame, "Y", (origin_px + 5, y_end - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_axes(self, frame):
        """Disegna assi X e Y sulla mappa usando world_to_pixel."""
        axis_len = 2.0  # lunghezza assi in metri

        origin_px, origin_py = self.world_to_pixel(0, 0)
        x_tip_px, x_tip_py = self.world_to_pixel(axis_len, 0)
        y_tip_px, y_tip_py = self.world_to_pixel(0, axis_len)

        # Asse X (rosso)
        cv2.arrowedLine(frame, (origin_px, origin_py), (x_tip_px, x_tip_py),
                       (0, 0, 255), 2, tipLength=0.15)
        cv2.putText(frame, "X", (x_tip_px + 5, x_tip_py + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Asse Y (verde)
        cv2.arrowedLine(frame, (origin_px, origin_py), (y_tip_px, y_tip_py),
                       (0, 255, 0), 2, tipLength=0.15)
        cv2.putText(frame, "Y", (y_tip_px + 5, y_tip_py - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def draw_trail(self, frame):
        """Disegna la traccia del percorso."""
        for trail, base_color in [(self.trail, (100, 150, 255)), (self.robot_trail, (100, 100, 100))]:
            if len(trail) < 2: continue
            for i in range(1, len(trail)):
                alpha = float(i) / len(trail)
                color = tuple(int(c * alpha) for c in base_color)
                p1 = self.world_to_pixel(trail[i-1][0], trail[i-1][1])
                p2 = self.world_to_pixel(trail[i][0], trail[i][1])
                if (0 <= p1[0] < self.window_size and 0 <= p1[1] < self.window_size and
                    0 <= p2[0] < self.window_size and 0 <= p2[1] < self.window_size):
                    cv2.line(frame, p1, p2, color, 2)

    def draw_human_marker(self, frame):
        """Disegna il marker dell'umano."""
        if not self.human_found:
            return

        px, py = self.world_to_pixel(self.human_x, self.human_y)

        if not (0 <= px < self.window_size and 0 <= py < self.window_size):
            return

        marker_size = int(0.5 * self.scale)
        marker_size = max(10, min(marker_size, 30))

        cv2.circle(frame, (px, py), marker_size, (0, 200, 255), -1)
        cv2.circle(frame, (px, py), marker_size, (0, 100, 200), 2)

        arrow_length = marker_size * 1.5
        # Direzione yaw in mondo: (cos(yaw), sin(yaw))
        # Applica stessa rotazione 90° antioraria di world_to_pixel: (dx,dy)->(-dy,-dx)
        dx = math.cos(self.human_yaw)
        dy = math.sin(self.human_yaw)
        end_x = int(px + arrow_length * (-dy))
        end_y = int(py + arrow_length * (-dx))

        cv2.arrowedLine(frame, (px, py), (end_x, end_y), (0, 0, 255), 3, tipLength=0.4)

    def draw_robot_marker(self, frame):
        """Disegna il marker del robot obstacle (cilindro nero)."""
        if not self.robot_found: return
        px, py = self.world_to_pixel(self.robot_x, self.robot_y)
        if not (0 <= px < self.window_size and 0 <= py < self.window_size): return
        marker_size = int(0.3 * self.scale)
        marker_size = max(8, min(marker_size, 25))
        cv2.circle(frame, (px, py), marker_size, (40, 40, 40), -1)
        cv2.circle(frame, (px, py), marker_size, (100, 100, 100), 2)
        # Freccia direzione
        arrow_length = marker_size * 1.3
        dx, dy = math.cos(self.robot_yaw), math.sin(self.robot_yaw)
        end_x, end_y = int(px + arrow_length * (-dy)), int(py + arrow_length * (-dx))
        cv2.arrowedLine(frame, (px, py), (end_x, end_y), (150, 150, 150), 2, tipLength=0.4)

    def draw_info(self, frame):
        """Disegna informazioni sulla mappa."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "MAPPA 2D", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.human_found:
            pos_text = "Pos: ({:.1f}, {:.1f}) m".format(self.human_x, self.human_y)
            cv2.putText(frame, pos_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            yaw_deg = math.degrees(self.human_yaw)
            yaw_text = "Yaw: {:.1f} deg".format(yaw_deg)
            cv2.putText(frame, yaw_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Umano non trovato", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        zoom_text = "Zoom: {:.0f}m".format(self.map_size_meters)
        cv2.putText(frame, zoom_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, "[+/-] Zoom  [C] Clear trail  [F] Follow  [ESC] Exit",
                   (10, self.window_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def draw_compass(self, frame):
        """Disegna bussola/indicatore nord."""
        cx = self.window_size - 40
        cy = 40
        radius = 25

        cv2.circle(frame, (cx, cy), radius, (100, 100, 100), 1)
        cv2.putText(frame, "N", (cx - 5, cy - radius - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.line(frame, (cx, cy), (cx, cy - radius + 5), (0, 255, 0), 2)
        cv2.putText(frame, "E", (cx + radius + 5, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def update_window(self):
        """Aggiorna la finestra."""
        if self.map_image is not None:
            frame = self.map_image.copy()
        else:
            frame = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            self.draw_grid(frame)

        self.draw_axes(frame)
        self.draw_trail(frame)
        self.draw_human_marker(frame)
        self.draw_robot_marker(frame)
        self.draw_compass(frame)
        self.draw_info(frame)

        cv2.imshow('Mappa 2D', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            rospy.loginfo('Chiusura...')
            rospy.signal_shutdown('User quit')
        elif key == ord('c'):
            self.trail = []
            self.robot_trail = []
            rospy.loginfo('Traccia cancellata')
        elif key == ord('+') or key == ord('='):
            self.map_size_meters = max(5.0, self.map_size_meters - 5.0)
            self.scale = self.window_size / self.map_size_meters
            rospy.loginfo('Zoom: {}m'.format(self.map_size_meters))
        elif key == ord('-'):
            self.map_size_meters = min(100.0, self.map_size_meters + 5.0)
            self.scale = self.window_size / self.map_size_meters
            rospy.loginfo('Zoom: {}m'.format(self.map_size_meters))
        elif key == ord('f'):
            if self.human_found:
                self.map_center_x = self.human_x
                self.map_center_y = self.human_y
                rospy.loginfo('Centrato su umano')
        elif key == ord('h'):
            self.map_center_x = 0.0
            self.map_center_y = 0.0
            rospy.loginfo('Centro reset')

    def run(self):
        """Loop principale."""
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.update_window()
            rate.sleep()


def main():
    print("\n" + "="*50)
    print("   VISUALIZZATORE MAPPA 2D")
    print("="*50)
    print("""
   Mostra la posizione dell'umano su una mappa
   vista dall'alto.

   Comandi:
     +/-  : Zoom in/out
     C    : Cancella traccia percorso
     F    : Centra su umano
     H    : Reset centro mappa
     Q/ESC: Esci
""")
    print("="*50 + "\n")

    node = MapViewer()
    try:
        node.run()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nMap Viewer terminato.")


if __name__ == "__main__":
    main()
