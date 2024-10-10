# coding:utf-8
"""
Raspberry Pi WiFi Wireless Video Car Robot Driver Source Code
Author: Sence
Copyright: XiaoR Technology (Shenzhen XiaoR Geek Technology Co., Ltd www.xiao-r.com); WiFi Robot Forum www.wifi-robots.com
This code can be freely modified, but it is forbidden to be used for commercial profit!
This code has been registered for software copyright protection. Any infringement found will result in immediate legal action!
"""
"""
@version: python3.7
@Author  : xiaor
@Explain : Camera recognition-related functions
@contact :
@Time    :2020/05/09
@File    :xr_camera.py
@Software: PyCharm
"""

from builtins import range, len, int
import os
from subprocess import call
import time
import math
import pyzbar.pyzbar as pyzbar
import xr_config as cfg

from xr_motor import RobotDirection

go = RobotDirection()
import cv2

from xr_servo import Servo

servo = Servo()

from xr_pid import PID


class Camera(object):
    def __init__(self):
        self.fre_count = 1  # Number of samples
        self.px_sum = 0  # Accumulated x-coordinate of sample points
        self.cap_open = 0  # Flag indicating whether the camera is open
        self.cap = None

        self.servo_X = 7
        self.servo_Y = 8

        self.angle_X = 90
        self.angle_Y = 20

        # Instantiate a PID algorithm for the X-axis. PID parameters: the first represents the P value, the second represents the I value, and the third represents the D value.
        self.X_pid = PID(0.03, 0.09, 0.0005)  
        self.X_pid.setSampleTime(0.005)  # Set the PID algorithm cycle
        self.X_pid.setPoint(240)  # Set the setpoint for the PID algorithm, i.e., the target value. Here 160 represents the center of the screen frame's x-axis. The x-axis has 320 pixels, half of which is 160.

        # Instantiate a PID algorithm for the Y-axis.
        self.Y_pid = PID(0.035, 0.08, 0.002)
        self.Y_pid.setSampleTime(0.005)  # Set the PID algorithm cycle
        self.Y_pid.setPoint(160)  # Set the setpoint for the PID algorithm, i.e., the target value. Here 160 represents the center of the screen frame's y-axis. The y-axis has 320 pixels, half of which is 160.

    def linepatrol_processing(self):
        """
        Camera line patrol data collection
        :return:
        """
        while True:
            if self.cap_open == 0:  # The camera is not open
                try:
                    # self.cap = cv2.VideoCapture(0) # Open the camera
                    self.cap = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream')
                except Exception as e:
                    print('opencv camera open error:', e)
                self.cap_open = 1  # Set the flag to 1
            else:
                try:
                    ret, frame = self.cap.read()  # Get a frame of data from the camera
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert RGB to GRAY color space
                        if cfg.PATH_DECT_FLAG == 0:
                            ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # Detect black lines
                        else:
                            ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # Detect white lines
                        for j in range(0, 640, 5):  # Sample points along the X-axis with a 5-pixel interval
                            if thresh1[350, j] == 0:  # Sample the points along the y-axis at 350, and apply binary thresholding
                                self.px_sum += j  # Accumulate the x-coordinates of the sampled points that match the line color
                                self.fre_count += 1  # Increase the sample count
                        cfg.LINE_POINT_ONE = self.px_sum / self.fre_count  # Calculate the average x-coordinate of the matching points (line position)
                        self.px_sum = 0  # Clear the accumulated value
                        self.fre_count = 1  # Reset the sample count to 1
                        for j in range(0, 640, 5):  # Sample points along the X-axis with a 5-pixel interval
                            if thresh1[200, j] == 0:  # Sample the points along the y-axis at 200, and apply binary thresholding
                                self.px_sum += j  # Accumulate the x-coordinates of the sampled points that match the line color
                                self.fre_count += 1  # Increase the sample count
                        cfg.LINE_POINT_TWO = self.px_sum / self.fre_count  # Calculate the average x-coordinate of the matching points (line position)
                        self.px_sum = 0  # Clear the accumulated value
                        self.fre_count = 1  # Reset the sample count to 1
                        print("point1 = %d ,point2 = %d"%(cfg.LINE_POINT_ONE,cfg.LINE_POINT_TWO))
                except Exception as e:  # Catch and print error messages
                    go.stop()  # Stop the robot
                    self.cap_open = 0  # Close the flag
                    self.cap.release()  # Release the camera
                    print('linepatrol_processing error:', e)

            if self.cap_open == 1 and cfg.CAMERA_MOD == 0:  # Exit line patrol mode
                go.stop()  # Stop the robot
                self.cap_open = 0  # Close the flag
                self.cap.release()  # Release the camera
                break  # Exit the loop

    def facefollow(self):
        """
        Face detection and camera following
        :return:
        """
        time.sleep(3)
        while True:

            if self.cap_open == 0:  # The camera is not open
                try:
                    # self.cap = cv2.VideoCapture(0)  # Open the camera
                    self.cap = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream')
                    self.cap_open = 1  # Set the flag to 1
                    self.cap.set(3, 320)  # Set the image width to 320 pixels
                    self.cap.set(4, 320)  # Set the image height to 320 pixels
                    face_cascade = cv2.CascadeClassifier(
                        '/home/pi/work/python_src/face.xml')  # Use OpenCV's face recognition cascade classifier, or switch to other feature detectors like for nose
                except Exception as e:
                    print('opencv camera open error:', e)
                    break

            else:
                try:
                    ret, frame = self.cap.read()  # Get the camera video stream
                    if ret == 1:  # Check if the camera is working
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert each frame to grayscale for face detection
                        faces = face_cascade.detectMultiScale(gray)  # Detect faces
                        if len(faces) > 0:  # If faces are detected
                            print('face found!')
                            for (x, y, w, h) in faces:
                                # Parameters are "target frame", "rectangle", "rectangle size", "line color", "width"
                                cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)
                                result = (x, y, w, h)
                                x_middle = result[0] + w / 2  # X-axis center
                                y_middle = result[1] + h / 2  # Y-axis center

                                self.X_pid.update(x_middle)  # Put X-axis data into the PID for output calculation
                                self.Y_pid.update(y_middle)  # Put Y-axis data into the PID for output calculation
                                self.angle_X = math.ceil(self.angle_X + 1 * self.X_pid.output)  # Update the X-axis servo angle
                                self.angle_Y = math.ceil(self.angle_Y + 0.8 * self.Y_pid.output)  # Update the Y-axis servo angle
                                self.angle_X = min(180, max(0, self.angle_X))  # Limit the X-axis angle
                                self.angle_Y = min(180, max(0, self.angle_Y))  # Limit the Y-axis angle
                                print("angle_X: %d" % self.angle_X)  # Print X-axis servo angle
                                print("angle_Y: %d" % self.angle_Y)  # Print Y-axis servo angle
                                servo.set(self.servo_X, self.angle_X)  # Set the X-axis servo
                                servo.set(self.servo_Y, self.angle_Y)  # Set the Y-axis servo
                except Exception as e:  # Catch and print error messages
                    go.stop()  # Stop the robot
                    self.cap_open = 0  # Close the flag
                    self.cap.release()  # Release the camera
                    print('facefollow error:', e)

            if self.cap_open == 1 and cfg.CAMERA_MOD == 0:  # If face detection mode ends, close the camera
                go.stop()  # Stop the robot
                self.cap_open = 0  # Close the flag
                self.cap.release()  # Release the camera
                break  # Exit the loop

    def run(self):
        """
        摄像头模式切换
        :return:
        """
        while True:
            if cfg.CAMERA_MOD == 1:  # 摄像头巡线模式
                cfg.LASRT_LEFT_SPEED = cfg.LEFT_SPEED  # 将当前速度保存
                cfg.LASRT_RIGHT_SPEED = cfg.RIGHT_SPEED
                cfg.LEFT_SPEED = 45  # 摄像头巡线时速度调低
                cfg.RIGHT_SPEED = 45
                self.linepatrol_processing()
            elif cfg.CAMERA_MOD == 2:  # 摄像头人脸检测跟随
                self.facefollow()
            elif cfg.CAMERA_MOD == 3:  # 摄像头颜色检测跟随
                self.colorfollow()
            elif cfg.CAMERA_MOD == 4:  # 摄像头二维码检测
                self.qrcode_detection()
            else:
                pass
            time.sleep(0.05)
