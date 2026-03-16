#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Actor Controller WASD - Due Modalita' di Controllo
Gazebo 11 + ROS 1 Noetic + ActorControlPlugin

Premi TAB per switchare tra le modalita'!

MODALITA' 1 - RELATIVO (stile videogioco 3D):
  W = Avanti (dove guarda)
  S = Indietro
  A = Ruota sinistra
  D = Ruota destra

MODALITA' 2 - ASSOLUTO (stile top-down):
  W = +Y (su)
  S = -Y (giu')
  A = -X (sinistra)
  D = +X (destra)

Animazioni (quando fermo):
  E = Talk A
  Q = Talk B

Altro:
  TAB = Cambia modalita'
  +/- = Cambia velocita'
  X/ESC = Esci

Requisiti:
  pip3 install pynput
"""

import sys
import math

try:
    from pynput import keyboard
except ImportError:
    print("Errore: Installa pynput: pip3 install pynput")
    sys.exit(1)

try:
    import rospy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
except ImportError:
    print("Errore: ROS 1 non trovato.")
    print("   source /opt/ros/noetic/setup.bash")
    sys.exit(1)


class WASDController(object):
    """Controller WASD con due modalita'."""

    # Modalita'
    MODE_RELATIVE = 0  # Stile videogioco 3D
    MODE_ABSOLUTE = 1  # Stile top-down
    MODE_NAMES = ["RELATIVO", "ASSOLUTO"]

    def __init__(self):
        rospy.init_node('wasd_controller', anonymous=False)

        # Publishers
        self.vel_pub = rospy.Publisher('/actor/cmd_vel', Twist, queue_size=10)
        self.anim_pub = rospy.Publisher('/actor/set_animation', String, queue_size=10)

        # Subscriber per stato
        self.state_sub = rospy.Subscriber('/actor/state', String, self.state_callback)

        # Velocita'
        self.move_speed = 1.5  # m/s
        self.turn_speed = 2.5  # rad/s

        # Modalita' corrente (leggi parametro, default relativo)
        start_mode = rospy.get_param('~start_mode', 'relative')
        if start_mode.lower() == 'absolute':
            self.mode = self.MODE_ABSOLUTE
        else:
            self.mode = self.MODE_RELATIVE

        # Stato tasti
        self.keys = {
            'w': False,
            's': False,
            'a': False,
            'd': False
        }

        self.current_state = "idle:walking"
        self.running = True
        self.is_moving = False

        # Timer a 60 Hz
        self.timer = rospy.Timer(rospy.Duration(1.0/60.0), self.update)

        rospy.loginfo('WASD Controller avviato!')

    def state_callback(self, msg):
        """Callback per stato attore."""
        self.current_state = msg.data

    def switch_mode(self):
        """Cambia modalita' di controllo."""
        self.mode = (self.mode + 1) % 2
        print("\n   Modalita': {}".format(self.MODE_NAMES[self.mode]))

    def update(self, event=None):
        """Pubblica velocita' basata sui tasti."""
        if not self.running:
            return

        msg = Twist()

        if self.mode == self.MODE_RELATIVE:
            # MODALITA' RELATIVA - Stile videogioco 3D
            forward = 0.0
            turn = 0.0

            if self.keys['w']:
                forward += 1.0
            if self.keys['s']:
                forward -= 1.0
            if self.keys['a']:
                turn += 1.0
            if self.keys['d']:
                turn -= 1.0

            self.is_moving = (abs(forward) > 0.01 or abs(turn) > 0.01)

            msg.linear.x = forward * self.move_speed
            msg.linear.y = 0.0
            msg.angular.z = turn * self.turn_speed

            self.print_status_relative(forward, turn)

        else:
            # MODALITA' ASSOLUTA - Stile top-down
            dx = 0.0
            dy = 0.0

            if self.keys['w']:
                dx += 1.0
            if self.keys['s']:
                dx -= 1.0
            if self.keys['a']:
                dy += 1.0
            if self.keys['d']:
                dy -= 1.0

            # Normalizza per movimento diagonale
            magnitude = math.sqrt(dx*dx + dy*dy)
            if magnitude > 0:
                dx /= magnitude
                dy /= magnitude

            self.is_moving = (magnitude > 0)

            msg.linear.x = dx * self.move_speed
            msg.linear.y = dy * self.move_speed
            msg.angular.z = 999.0  # Flag: modalita' assoluta

            self.print_status_absolute(dx, dy)

        self.vel_pub.publish(msg)

    def print_status_relative(self, forward, turn):
        """Stampa stato modalita' relativa."""
        if forward > 0:
            arrow = '^' if turn == 0 else ('<' if turn > 0 else '>')
        elif forward < 0:
            arrow = 'v' if turn == 0 else ('<' if turn > 0 else '>')
        elif turn > 0:
            arrow = '<'
        elif turn < 0:
            arrow = '>'
        else:
            arrow = '.'

        self._print_line(arrow, "REL")

    def print_status_absolute(self, dx, dy):
        """Stampa stato modalita' assoluta."""
        arrows = {
            (0, 1): '^', (0, -1): 'v',
            (-1, 0): '<', (1, 0): '>',
            (-1, 1): '\\', (1, 1): '/',
            (-1, -1): '/', (1, -1): '\\',
            (0, 0): '.'
        }
        key = (int(round(dx)), int(round(dy)))
        arrow = arrows.get(key, '.')

        self._print_line(arrow, "ABS")

    def _print_line(self, arrow, mode_short):
        """Stampa linea di stato."""
        pressed = ''.join(k.upper() for k, v in self.keys.items() if v) or '-'

        state_parts = self.current_state.split(':')
        anim_name = state_parts[1] if len(state_parts) > 1 else '?'

        if self.is_moving:
            status = "[{}]".format(arrow)
            info = "v={:.1f}".format(self.move_speed)
        else:
            status = "[{}]".format(arrow)
            info = "[{}]".format(anim_name[:8])

        line = "\r{} | {} | [{}] | {} | TAB=mode E/Q=talk  ".format(
            status, mode_short, pressed.ljust(4), info.ljust(10))
        sys.stdout.write(line)
        sys.stdout.flush()

    def play_animation(self, anim_name):
        """Riproduce un'animazione."""
        if self.is_moving:
            return

        msg = String()
        msg.data = anim_name
        self.anim_pub.publish(msg)
        print("\n   Animazione: {}".format(anim_name))

    def on_press(self, key):
        """Tasto premuto."""
        try:
            # Tab per cambiare modalita'
            if key == keyboard.Key.tab:
                self.switch_mode()
                return

            if hasattr(key, 'char') and key.char:
                k = key.char.lower()
                if k in self.keys:
                    self.keys[k] = True
                elif k == '+' or k == '=':
                    self.move_speed = min(5.0, self.move_speed + 0.25)
                    print("\n   Velocita': {:.2f} m/s".format(self.move_speed))
                elif k == '-':
                    self.move_speed = max(0.5, self.move_speed - 0.25)
                    print("\n   Velocita': {:.2f} m/s".format(self.move_speed))
                elif k == 'e':
                    self.play_animation('talk_a')
                elif k == 'q':
                    self.play_animation('talk_b')
        except Exception:
            pass

    def on_release(self, key):
        """Tasto rilasciato."""
        try:
            if hasattr(key, 'char') and key.char:
                k = key.char.lower()
                if k in self.keys:
                    self.keys[k] = False
                elif k == 'x':
                    self.running = False
                    return False
        except Exception:
            pass
        if key == keyboard.Key.esc:
            self.running = False
            return False

    def run(self):
        """Avvia controller."""
        self._print_header()

        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()

        try:
            rate = rospy.Rate(100)
            while self.running and not rospy.is_shutdown():
                rate.sleep()
        except (KeyboardInterrupt, rospy.ROSInterruptException):
            pass
        finally:
            self.vel_pub.publish(Twist())
            listener.stop()
            print("\n\nController terminato.")

    def _print_header(self):
        """Stampa header."""
        print("\n" + "="*60)
        print("   WASD CONTROLLER - Due Modalita'")
        print("="*60)
        print("""
   TAB = Cambia modalita'

   MODALITA' RELATIVO (corrente):     MODALITA' ASSOLUTO:
   W = Avanti (dove guarda)           W = +Y (su nel mondo)
   S = Indietro                       S = -Y (giu')
   A = Ruota sinistra                 A = -X (sinistra)
   D = Ruota destra                   D = +X (destra)

   Animazioni:  E = Talk A    Q = Talk B
   Velocita':   +/- per cambiare
   Esci:        X o ESC
""")
        print("   Modalita': {}".format(self.MODE_NAMES[self.mode]))
        print("   Velocita': {} m/s".format(self.move_speed))
        print("\n" + "="*60 + "\n")


def main():
    node = WASDController()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
