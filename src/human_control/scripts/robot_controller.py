#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controller Frecce per Robot Obstacle (cilindro nero).
Pubblica su /robot/cmd_vel - il plugin Gazebo gestisce collisioni.

Comandi: Frecce=movimento, +/-=velocità, ESC/X=esci
"""
import sys
import math
try:
    from pynput import keyboard
except ImportError:
    print("pip3 install pynput"); sys.exit(1)

import rospy
from geometry_msgs.msg import Twist

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=False)
        self.vel_pub = rospy.Publisher('/robot/cmd_vel', Twist, queue_size=10)
        self.speed = 1.5
        self.keys = {'up': False, 'down': False, 'left': False, 'right': False}
        self.running = True
        self.timer = rospy.Timer(rospy.Duration(1.0/60.0), self.update)
        rospy.loginfo('Robot Controller avviato! Frecce=muovi, +/-=velocità, X/ESC=esci')

    def update(self, event=None):
        if not self.running: return
        dx = dy = 0.0
        if self.keys['up']: dx += 1.0
        if self.keys['down']: dx -= 1.0
        if self.keys['left']: dy += 1.0
        if self.keys['right']: dy -= 1.0
        mag = math.sqrt(dx*dx + dy*dy)
        if mag > 0: dx, dy = dx/mag, dy/mag
        msg = Twist()
        msg.linear.x = dx * self.speed
        msg.linear.y = dy * self.speed
        self.vel_pub.publish(msg)

    def on_press(self, key):
        try:
            # Gestione frecce direzionali
            if key == keyboard.Key.up: self.keys['up'] = True
            elif key == keyboard.Key.down: self.keys['down'] = True
            elif key == keyboard.Key.left: self.keys['left'] = True
            elif key == keyboard.Key.right: self.keys['right'] = True
            # Gestione velocità e uscita
            elif hasattr(key, 'char') and key.char:
                k = key.char.lower()
                if k in ['+', '=']: self.speed = min(5.0, self.speed + 0.25); print("\n  Robot Vel: {:.2f}".format(self.speed))
                elif k == '-': self.speed = max(0.5, self.speed - 0.25); print("\n  Robot Vel: {:.2f}".format(self.speed))
                elif k == 'x': self.running = False; return False
        except: pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.up: self.keys['up'] = False
            elif key == keyboard.Key.down: self.keys['down'] = False
            elif key == keyboard.Key.left: self.keys['left'] = False
            elif key == keyboard.Key.right: self.keys['right'] = False
        except: pass
        if key == keyboard.Key.esc: self.running = False; return False

    def run(self):
        print("\n" + "="*40 + "\n  ROBOT CONTROLLER\n  Frecce=muovi +/-=vel X/ESC=esci\n" + "="*40)
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        rate = rospy.Rate(100)
        while self.running and not rospy.is_shutdown(): rate.sleep()
        self.vel_pub.publish(Twist())  # Stop
        listener.stop()
        print("\nTerminato.")

if __name__ == "__main__":
    RobotController().run()
