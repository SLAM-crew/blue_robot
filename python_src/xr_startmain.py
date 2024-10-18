# coding:utf-8

import os
import time
import threading
from threading import Timer
from solution.manipulator import init_pose
from solution.team_light import turn_on_red_light_via_i2c
import xr_config as cfg
from xr_oled import Oled
try:
    oled = Oled()
except:
    print('oled initialization fail')
from xr_power import Power
power = Power()


def status():
	if cfg.PROGRAM_ABLE:
		if cfg.LOOPS > 100: 
			cfg.LOOPS = 0
			# power.show_vol() 
			try:
				oled.disp_cruising_mode()
			except:
				print('oled initialization fail')

	loops = cfg.LOOPS
	loops = loops + 1
	cfg.LOOPS = loops 

	Timer(0.01, status).start()


if __name__ == '__main__':
	init_pose()
	turn_on_red_light_via_i2c()
	print("....wifirobots start!...")

	os.system("sudo hciconfig hci0 name XiaoRGEEK")
	time.sleep(0.1)
	os.system("sudo hciconfig hci0 reset")
	time.sleep(0.3)
	os.system("sudo hciconfig hci0 piscan")
	time.sleep(0.2)
	print("now bluetooth discoverable")
	try:
		oled.disp_default()	
	except:
		print('oled initialization fail')

ti = threading.Timer(0.1, status)
ti.start()