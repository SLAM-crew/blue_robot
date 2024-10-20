import xr_gpio as gpio
import time
from  collections import deque
def get_ir():
    buf = [gpio.digital_read(gpio.IR_L),
    gpio.digital_read(gpio.IR_R),
    gpio.digital_read(gpio.IR_M),
    gpio.digital_read(gpio.IRF_L),
    gpio.digital_read(gpio.IRF_R)]
    return buf

def get_distance():
	time_count = 0
	time.sleep(0.01)
	gpio.digital_write(gpio.TRIG, True) 
	time.sleep(0.000015)
	gpio.digital_write(gpio.TRIG, False) 
	while not gpio.digital_read(gpio.ECHO): 
		pass
	t1 = time.time()  
	while gpio.digital_read(gpio.ECHO): 
		if time_count < 2000: 
			time_count = time_count + 1
			time.sleep(0.000001)
			pass
		else:
			print("NO ECHO receive! Please check connection")
			break
	t2 = time.time() 
	distance = (t2 - t1) * 340 / 2 * 100  

	return distance


def moving_averages(distance):
    window_size = 5
    readings = deque(maxlen=5)
    curr_sum = 0.0
    if len(readings) == window_size:
        curr_sum -= readings[0]
    readings.append(distance)
    curr_sum += distance
    if len(readings) == 0:
        return 0
    return round(curr_sum/len(readings), 2)
