#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualizzatore Vista Prima Persona
Mostra il flusso video dalla camera dell'umano in Gazebo.

Requisiti:
  pip3 install opencv-python

Uso:
  rosrun human_control fpv_viewer.py
"""

import sys

try:
    import cv2
    import numpy as np
except ImportError:
    print("Errore: Installa OpenCV: pip3 install opencv-python")
    sys.exit(1)

try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
except ImportError:
    print("Errore: ROS 1 non trovato.")
    print("   source /opt/ros/noetic/setup.bash")
    sys.exit(1)


class FPVViewer(object):
    """Visualizzatore vista prima persona."""

    def __init__(self):
        rospy.init_node('fpv_viewer', anonymous=False)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_count = 0

        # Subscriber per l'immagine della camera
        self.image_sub = rospy.Subscriber(
            '/first_person_camera/image_raw',
            Image,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo('FPV Viewer avviato!')
        rospy.loginfo('Topic: /first_person_camera/image_raw')
        rospy.loginfo('Premi ESC o Q per uscire')

        # Crea finestra
        cv2.namedWindow('Vista Prima Persona', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vista Prima Persona', 800, 600)

    def image_callback(self, msg):
        """Callback per nuova immagine."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
        except Exception as e:
            rospy.logerr('Errore conversione immagine: {}'.format(e))

    def update_window(self):
        """Aggiorna la finestra."""
        if self.latest_frame is not None:
            frame = self.latest_frame.copy()

            # Testo info
            info_text = "FPV Camera | Frame: {}".format(self.frame_count)
            cv2.putText(frame, info_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crosshair al centro
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)

            cv2.imshow('Vista Prima Persona', frame)
        else:
            waiting = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting, "In attesa del flusso video...",
                       (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(waiting, "Topic: /first_person_camera/image_raw",
                       (110, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            cv2.imshow('Vista Prima Persona', waiting)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            rospy.loginfo('Chiusura...')
            rospy.signal_shutdown('User quit')

    def run(self):
        """Loop principale."""
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            self.update_window()
            rate.sleep()


def main():
    print("\n" + "="*50)
    print("   VISTA PRIMA PERSONA")
    print("="*50)
    print("""
   Visualizza il flusso video dalla camera
   posizionata agli "occhi" dell'umano.

   Premi ESC o Q per uscire.
""")
    print("="*50 + "\n")

    node = FPVViewer()
    try:
        node.run()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nViewer terminato.")


if __name__ == "__main__":
    main()
