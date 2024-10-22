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

# CHANGE BEFORE START
GREEN = False
ROBOT_IP = '192.168.2.12' # 192.168.2.53
DIRECTION = "S"

RECORD = False
TELEOP = False
K = 0.75

cube_hunt, ball_hunt = False, False
init = True
flag = None
trajectory = []
recover_angle = None
app = Flask(__name__)

@app.route('/trajectory_points', methods=['POST'])
def trajectory_handler():
    global trajectory, flag
    try:
        data = request.json
        received_string = data.get('trajectory')

        if received_string:
            trajectory = received_string
            flag = data.get('flag')

            return jsonify({"status": "success", "message": "trajectory recieved"}), 200
        else:
            return jsonify({"status": "error", "message": "No trajectory received"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/recover_angle', methods=['POST'])
def recover_angle_handler():
    global recover_angle
    try:
        data = request.json
        received_string = [data.get('axis'), data.get('angle')]

        if received_string:
            recover_angle = received_string
            return jsonify({"status": "success", "message": "recover angle recieved"}), 200
        else:
            return jsonify({"status": "error", "message": "No recover angle received"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

class Solution():
    def __init__(self):
        stop()
        if GREEN:
            turn_on_green_light_via_i2c()
        else:
            turn_on_red_light_via_i2c()
        init_pose()

        self.linear_velocity = 0.1175 
        self.angular_velocity = 0.79 #0.62 #0.76 
        self.motor_pwm = 25

        self.direction = DIRECTION

        self.pose = None

        self.pid = PID(0.042, 0, 0)
        self.pid.setSampleTime(0.075)
        self.pid.setPoint(320)
        
        self.cap = cv2.VideoCapture('/dev/video0')
        if RECORD:
            self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 30, (640,240))
        signal.signal(signal.SIGINT, self.shutdown)
        print('Init')

    def shutdown(self, sig, frame):
        # if GREEN:
        #     turn_on_red_light_via_i2c()
        # else:
        #     turn_on_green_light_via_i2c()
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
        # while abs(current_time - last_time) < dt:
        #     current_time = time.time()
        time.sleep(dt)
        stop()
        time.sleep(0.5)

    def move_forward(self, distance):
        dt = distance / self.linear_velocity
        current_time = time.time()
        last_time = current_time
        set_velocities(self.motor_pwm, K * self.motor_pwm)
        # while abs(current_time - last_time) < dt:
        #     current_time = time.time()
        time.sleep(dt)
        stop()
        time.sleep(0.5)


    def back_obstacles(self, left_ik, right_ik):
        stop()
        while not left_ik or not right_ik:
            if not left_ik:
                self.rotate(-np.pi/2)
            elif not right_ik:
                self.rotate(np.pi/2)
            left_ik = get_ir()[0]
            right_ik = get_ir()[1]
        stop()

     
    def spin(self):
        global trajectory, cube_hunt, recover_angle, ball_hunt, flag, init
        while True:
            # start_time = time.time()
            # left_ik = get_ir()[0]
            # right_ik = get_ir()[1]
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
                        if target[1] < self.pose[1]:
                            self.rotate(np.pi)
                            self.direction = 'W'
                        else:
                            pass
                        dst = abs(target[1] - self.pose[1])
                self.move_forward(dst / 100)

            else:
                if flag is not None:
                    if flag == 'cube':
                        cube_hunt = True
                    elif flag == 'ball':
                        ball_hunt = True
                    flag = None

            if recover_angle is not None:
                self.rotate(recover_angle[1])
                self.direction = recover_angle[0]
                recover_angle = None

            # if not left_ik or not right_ik:
            #     self.back_obstacles(left_ik, right_ik)
            #     continue

            if ball_hunt:
                if init:
                    set_velocities(-self.motor_pwm, -self.motor_pwm)
                    time.sleep(1.5)
                    stop()
                    init = False
                ir = get_ir()
                if ir[2] == 0:
                    stop()
                    grab_item(0.16)
                    init_pose()
                    ball_hunt = False
                    continue
                _, frame = self.cap.read()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                y, x, _ = frame.shape
                # frame = frame[int(0.5 * y):y, int(0.25 * x):x - int(0.25 * x)]
                frame = frame[int(0.5 * y):y, 0:x]
                # frame = align_histogram(frame)
                # frame = white_balance(frame)
                if RECORD:
                    self.out.write(frame)
                contour = apply_mask(frame)
                if  contour is not None:
                    x, _ = object_center(contour)
                    self.pid.update(x)
                    # print(self.pid.output)
                    left_vel = self.motor_pwm -  2 * self.pid.output
                    right_vel = self.motor_pwm + 2 * self.pid.output
                    set_velocities(left_vel, K * right_vel)  
                    # print(time.time() - start_time)
                else:
                    self.rotate(np.pi / 6)

            if cube_hunt:
                if init:
                    set_velocities(-self.motor_pwm, -self.motor_pwm)
                    time.sleep(1.5)
                    stop()
                    init = False
                ir = get_ir()
                if ir[2] == 0:
                    stop()
                    grab_item(0.17)
                    init_pose()
                    cube_hunt = False
                    continue
                _, frame = self.cap.read()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                y, x, _ = frame.shape
                # frame = frame[int(0.5 * y):y, int(0.25 * x):x - int(0.25 * x)]
                frame = frame[int(0.5 * y):y, 0:x]
                # frame = align_histogram(frame)
                # frame = white_balance(frame)
                if RECORD:
                    self.out.write(frame)
                contour = apply_mask(frame)
                if  contour is not None:
                    x, _ = object_center(contour)
                    self.pid.update(x)
                    # print(self.pid.output)
                    left_vel = self.motor_pwm -  2 * self.pid.output
                    right_vel = self.motor_pwm + 2 * self.pid.output
                    set_velocities(left_vel, K * right_vel)  
                    # print(time.time() - start_time)      
                else:
                    self.rotate(np.pi / 6)
      

if __name__ == "__main__":
    if TELEOP:
        curses.wrapper(teleop)
    else:
        sol = Solution()
        threading.Thread(target= lambda: app.run(host=ROBOT_IP, port=5000, debug=False)).start()
        # sol.rotate(np.pi / 2)
        # sol.move_forward(1)
        sol.spin()