import xr_gpio as gpio

def m1m2_forward():
    gpio.digital_write(gpio.IN1, True)
    gpio.digital_write(gpio.IN2, False)

def m1m2_reverse():
    gpio.digital_write(gpio.IN1, False)
    gpio.digital_write(gpio.IN2, True)

def m1m2_stop():
    gpio.digital_write(gpio.IN1, False)
    gpio.digital_write(gpio.IN2, False)

def m3m4_forward():
    gpio.digital_write(gpio.IN3, True)
    gpio.digital_write(gpio.IN4, False)

def m3m4_reverse():
    gpio.digital_write(gpio.IN3, False)
    gpio.digital_write(gpio.IN4, True)

def m3m4_stop():
    gpio.digital_write(gpio.IN3, False)
    gpio.digital_write(gpio.IN4, False)

def set_speed(num, speed):
    if num == 1:
        gpio.ena_pwm(speed)
    elif num == 2:
        gpio.enb_pwm(speed)

def stop():
    set_speed(1, 0)
    set_speed(2, 0)
    m1m2_stop()
    m3m4_stop()

def set_velocities(speed_left, speed_right):
    
    if speed_left > 0: 
        gpio.ena_pwm(speed_left)
        m1m2_forward()
    else:
        gpio.ena_pwm(-speed_left)
        m1m2_reverse()
    if speed_right > 0: 
        gpio.enb_pwm(speed_right)
        m3m4_forward()
    else:
        gpio.enb_pwm(-speed_right)
        m3m4_reverse()