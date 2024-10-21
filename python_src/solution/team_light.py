import time
from xr_i2c import I2c

i2c = I2c()

def turn_on_red_light_via_i2c():
    """
    Включить все светодиоды красным цветом напрямую через I2C
    """
    i2c_address = i2c.mcu_address  # Адрес устройства на шине I2C
    group_1 = 1  # Группа светодиодов (2 для car light)
    group_2 = 2
    color_red = 1  # Код красного цвета (предположительно 1 для красного)
    num_leds = 8  # Количество светодиодов

    # Цикл для включения каждого светодиода в группе
    for led_num in range(1, num_leds + 1):
        sendbuf = [0xff, group_1 + 3, led_num, color_red, 0xff]  # Формируем команду
        i2c.writedata(i2c_address, sendbuf)  # Отправляем данные через I2C
        sendbuf = [0xff, group_2 + 3, led_num, color_red, 0xff]  # Формируем команду
        i2c.writedata(i2c_address, sendbuf)  # Отправляем данные через I2C
        time.sleep(0.005)  # Небольшая задержка между включениями

def turn_on_green_light_via_i2c():
    """
    Включить все светодиоды красным цветом напрямую через I2C
    """
    i2c_address = i2c.mcu_address  # Адрес устройства на шине I2C
    group_1 = 1  # Группа светодиодов (2 для car light)
    group_2 = 2
    color_green = 4  # Код красного цвета (предположительно 1 для красного)
    num_leds = 8  # Количество светодиодов

    # Цикл для включения каждого светодиода в группе
    for led_num in range(1, num_leds + 1):
        sendbuf = [0xff, group_1 + 3, led_num, color_green, 0xff]  # Формируем команду
        i2c.writedata(i2c_address, sendbuf)  # Отправляем данные через I2C
        sendbuf = [0xff, group_2 + 3, led_num, color_green, 0xff]  # Формируем команду
        i2c.writedata(i2c_address, sendbuf)  # Отправляем данные через I2C
        time.sleep(0.005)  # Небольшая задержка между включениями

# turn_on_green_light_via_i2c()