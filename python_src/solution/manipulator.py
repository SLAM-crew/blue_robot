import time
import math
import numpy as np
from xr_i2c import I2c

i2c = I2c()


def reach_pose(x, z):
    l1 = 0.095
    l2 = 0.09

    b = np.sqrt(x ** 2 + z ** 2)
    a1 = -np.arccos((b ** 2 + l1 ** 2 - l2 ** 2) / (2 * l1 * b))
    A3 = np.arctan(z / x)
    A1 = A3 - a1
    a2 = -np.arccos((l1 ** 2 + l2 ** 2 - b ** 2) / (2 * l1 * l2))
    A2 = np.pi - a2

    sol = [np.pi / 2 - A1, 2 * np.pi - A2, 0]
    servo = [0x01, 0x04, 0x03]
    setpoint = [np.pi / 2, np.pi, np.pi / 2]
    for i in range(3):
        i2c.writedata(i2c.mcu_address, [0xff, 0x01, servo[i], int(math.degrees(setpoint[i] - sol[i])) , 0xff])
    time.sleep(0.5)

def end_effector(angle):
    i2c.writedata(i2c.mcu_address, [0xff, 0x01, 0x02, angle, 0xff])
    time.sleep(0.5)


def init_pose():
    reach_pose(0.09, 0.1)
    end_effector(75)

def grab_item(dst):
    end_effector(45)
    reach_pose(dst, -0.025)
    end_effector(75)

def put_item(dst):
    reach_pose(dst, 0)
    end_effector(45)

def push_button(dst):
    reach_pose(dst, 0.11 - 0.055)
