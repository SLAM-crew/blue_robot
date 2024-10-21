import cv2
import signal
import sys
import time
import curses
import numpy as np
from teleop import teleop
from cv import apply_mask, align_histogram, white_balance, object_center
from manipulator import init_pose, grab_item
from motors import set_velocities, stop
from team_light import turn_on_red_light_via_i2c, turn_on_green_light_via_i2c
from sensors import get_ir
from flask import Flask, request, jsonify
from xr_pid import PID
import threading

RECORD = True
TELEOP = False
GREEN = True
K = 0.84
CUBE_HUNT = False

trajectory = []
app = Flask(__name__)

def handle_received_trajectory(received_poins):
    global trajectory, CUBE_HUNT
    trajectory = received_poins
    CUBE_HUNT = True

@app.route('/trajectory_points', methods=['POST'])
def trajectory_handler():
    try:
        data = request.json
        received_string = data.get('trajectory')

        if received_string:
            response_message = handle_received_trajectory(received_string)
          
            return jsonify({"status": "success", "message": response_message}), 200
        else:
            return jsonify({"status": "error", "message": "No string received"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

class Solution():
    def __init__(self):
        if GREEN:
            turn_on_green_light_via_i2c()
        else:
            turn_on_red_light_via_i2c()
        init_pose()

        self.linear_velocity = 0.2 # 0.25
        self.angular_velocity = 1.4 # 1.58
        self.motor_pwm = 40

        self.direction = 'N'

        self.pose = None

        self.pid = PID(0.03, 0.09, 0.0005)
        self.pid.setSampleTime(0.005)
        self.pid.setPoint(320)
        
        self.cap = cv2.VideoCapture('/dev/video0')
        if RECORD:
            self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 30, (640,240))
        
        signal.signal(signal.SIGINT, self.shutdown)
        print('Init')

    def shutdown(self, sig, frame):
        if GREEN:
            turn_on_red_light_via_i2c()
        else:
            turn_on_green_light_via_i2c()
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
            current_time = time.time()
        stop()
    
    def move_forward(self, distance):
        dt = distance / self.linear_velocity
        current_time = time.time()
        last_time = current_time
        set_velocities(self.motor_pwm, K * self.motor_pwm)
        while abs(current_time - last_time) < dt:
            current_time = time.time()
        stop()

    def spin(self):
        global trajectory
        while True:
            if len(trajectory) > 1:
                self.pose = trajectory[0]
                trajectory.pop(0)
                target = trajectory[0]
                if self.direction == 'N':
                    if self.pose[0] == target[0]:
                        if target[1] < self.pose[1]:
                            self.rotate(-np.pi / 2)
                            self.direction = 'W'
                        else:
                            self.rotate(np.pi / 2)
                            self.direction = 'E'
                        dst = abs(target[1] - self.pose[1])
                    else:
                        if target[0] > self.pose[0]:
                            self.rotate(np.pi)
                            self.direction = 'S'
                        else:
                            pass
                        dst = abs(target[0] - self.pose[0])

                elif self.direction == 'S':
                    if self.pose[0] == target[0]:
                        if target[1] < self.pose[1]:
                            self.rotate(np.pi / 2)
                            self.direction = 'W'
                        else:
                            self.rotate(-np.pi / 2)
                            self.direction = 'E'
                        dst = abs(target[1] - self.pose[1])
                    else:
                        if target[0] < self.pose[0]:
                            self.rotate(np.pi)
                            self.direction = 'N'
                        else:
                            pass
                        dst = abs(target[0] - self.pose[0])


                elif self.direction == 'W':
                    if self.pose[1] == target[1]:
                        if target[0] < self.pose[0]:
                            self.rotate(np.pi / 2)
                            self.direction = 'N'
                        else:
                            self.rotate(-np.pi / 2)
                            self.direction = 'S'
                        dst = abs(target[0] - self.pose[0])
                    else:
                        if target[1] > self.pose[1]:
                            self.rotate(np.pi)
                            self.direction = 'E'
                        else:
                            pass
                        dst = abs(target[1] - self.pose[1])


                elif self.direction == 'E':
                    if self.pose[1] == target[1]:
                        if target[0] < self.pose[0]:
                            self.rotate(-np.pi / 2)
                            self.direction = 'N'
                        else:
                            self.rotate(np.pi / 2)
                            self.direction = 'S'
                        dst = abs(target[0] - self.pose[0])
                    else:
                        if target[1] > self.pose[1]:
                            self.rotate(np.pi)
                            self.direction = 'W'
                        else:
                            pass
                        dst = abs(target[1] - self.pose[1])
                self.move_forward(dst / 100)
            else:
                if CUBE_HUNT:
                    self.shutdown(None, None)
                    ir = get_ir()
                    if ir[2] == 0:
                        stop()
                        grab_item(0.14)
                        init_pose()
                        self.shutdown(None, None)
                    _, frame = self.cap.read()
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    y, x, _ = frame.shape
                    # frame = frame[int(0.5 * y):y, int(0.25 * x):x - int(0.25 * x)]
                    frame = frame[int(0.5 * y):y, 0:x]
                    frame = align_histogram(frame)
                    frame = white_balance(frame)
                    if RECORD:
                        self.out.write(frame)
                    contour = apply_mask(frame)
                    if  contour is not None:
                        x, _ = object_center(contour)
                        self.pid.update(x)
                        print(self.pid.output)
                        left_vel = self.motor_pwm -  2 * self.pid.output
                        right_vel = self.motor_pwm + 2 * self.pid.output
                        set_velocities(left_vel, K * right_vel)           

if __name__ == "__main__":
    if TELEOP:
        curses.wrapper(teleop)
    else:
        sol = Solution()
        threading.Thread(target= lambda: app.run(host='192.168.8.254', port=5000, debug=False)).start()
        sol.spin()