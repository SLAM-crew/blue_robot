import cv2
import signal
import sys
import time
import curses
import numpy as np
from cv import follow_cube, align_histogram, white_balance, undistored, get_distance
from manipulator import init_pose, grab_item, put_item
from motors import set_velocities, stop
from team_light import turn_on_red_light_via_i2c, turn_on_green_light_via_i2c
from sensors import get_ir
from xr_pid import PID

RECORD = False
TELEOP = False
GREEN = True

K = 0.84

def teleop(stdscr):
    if GREEN:
            turn_on_green_light_via_i2c()
    else:
        turn_on_red_light_via_i2c()
    init_pose()
    if RECORD:
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 30, (320, 240))
        cap = cv2.VideoCapture('/dev/video0')
    curses.cbreak()  
    stdscr.keypad(True) 
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.refresh()

    stdscr.addstr(0, 0, "Управляйте роботом с помощью клавиш W, A, S, D. Для выхода нажмите 'q'.")

    speed = 40
    
    while True:
        if RECORD:
            _, frame = cap.read()
            # frame = undistored(frame)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            y, x, _ = frame.shape
            frame = frame[int(0.5 * y):y, int(0.25 * x):x - int(0.25 * x)]
            frame = align_histogram(frame)
            frame = white_balance(frame)
            out.write(frame)
        key = stdscr.getch()
        
        if key == ord('q'):
            if RECORD:
                out.release()
                cap.release()
            break 
        
        if key == ord('+') and speed < 100:
            speed += 1
            stdscr.addstr(2, 0, f"Текущая скорость: {speed:3d}   ")
        
        elif key == ord('-') and speed > 0:
            speed -= 1 
            stdscr.addstr(2, 0, f"Текущая скорость: {speed:3d}   ")
        
        elif key != -1:
            if key == ord('w'):
                set_velocities(speed, K*speed)        
            elif key == ord('s'):
                set_velocities(-speed, -K*speed)   
            elif key == ord('a'):
                set_velocities(-speed, K*speed)
            elif key == ord('d'):
                set_velocities(speed, -K*speed)
            elif key == ord('x'):
                stop()
        time.sleep(0.1)

class Solution():
    def __init__(self):
        if GREEN:
            turn_on_green_light_via_i2c()
        else:
            turn_on_red_light_via_i2c()
        init_pose()
        self.linear_velocity = 0.25 
        self.angular_velocity = 1.58
        self.motor_pwm = 40
        self.cap = cv2.VideoCapture('/dev/video0')
        self.pid = PID(0.03, 0.09, 0.0005)
        self.pid.setSampleTime(0.005)
        self.pid.setPoint(320)
        if RECORD:
            self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 30, (320,240))
        signal.signal(signal.SIGINT, self.shutdown)
        print('Init')

    def shutdown(self, sig, frame):
        init_pose()
        stop()
        if RECORD:
            self.out.release()
        self.cap.release()
        print('Shutdown')
        sys.exit(0)
    
    def rotate(self, angle):
        sign = np.sign(angle)
        dt = abs(angle) / self.angular_velocity
        current_time = time.time()
        last_time = current_time
        set_velocities(sign * self.motor_pwm, -sign * K * self.motor_pwm)
        while abs(current_time - last_time) < dt:
            # print(current_time)
            current_time = time.time()
        stop()

    
    def move_forward(self, distance):
        dt = distance / self.linear_velocity
        current_time = time.time()
        last_time = current_time
        set_velocities(self.motor_pwm, K * self.motor_pwm)
        while abs(current_time - last_time) < dt:
            # print(current_time)
            current_time = time.time()
        stop()

    def spin(self):
        while True:
            base_vel = 35
            left_vel = base_vel
            right_vel = base_vel
            ir = get_ir()
            if ir[2] == 0:
                dst = 0.14
                stop()
                grab_item(dst)
                init_pose()
                time.sleep(2)
                put_item(dst)
                self.shutdown(None, None)

            _, frame = self.cap.read()
            # frame = undistored(frame)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            y, x, _ = frame.shape
            # frame = frame[int(0.5 * y):y, int(0.25 * x):x - int(0.25 * x)]
            frame = frame[int(0.5 * y):y, 0:x]
            frame = align_histogram(frame)
            frame = white_balance(frame)
            if RECORD:
                self.out.write(frame)
            res = follow_cube(frame)
            if  res is not None:
                x, _ = res
                self.pid.update(x)
                print(self.pid.output)
                left_vel -= 2 * self.pid.output
                right_vel += 2 * self.pid.output
            set_velocities(left_vel, right_vel)

if __name__ == "__main__":
    if TELEOP:
        curses.wrapper(teleop)
    else:
        sol = Solution()
        # sol.spin()
        sol.move_forward(1)
        time.sleep(0.5)
        sol.rotate(np.pi / 2)
        time.sleep(0.5)
        sol.move_forward(0.5)
        time.sleep(0.5)
        sol.rotate(np.pi / 2)
        time.sleep(0.5)
        sol.move_forward(0.25)
        time.sleep(0.5)
        sol.rotate(-np.pi / 2)
        time.sleep(0.5)
        sol.move_forward(2)