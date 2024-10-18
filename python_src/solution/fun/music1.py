import time
import xr_gpio as gpio


class Beep(object):
    def __init__(self):
        self.buzzer_pin = gpio.BUZZER
        

    def tone(self, frequency, duration):
        """Воспроизведение звука на указанной частоте и длительности"""
        if frequency == 0:  # Пауза
            time.sleep(duration / 1000)
        else:
            period = 1.0 / frequency
            delay_value = period / 2
            cycles = int(frequency * duration / 1000)
            for i in range(cycles):
                gpio.digital_write(self.buzzer_pin, True)
                time.sleep(delay_value)
                gpio.digital_write(self.buzzer_pin, False)
                time.sleep(delay_value)

    def was_wollen(self):
        # Пример нот с их частотой в герцах и длительностями в миллисекундах
        self.tone(1174, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1174, 108.841490625)
        time.sleep(0.12195125)
        self.tone(1046, 108.841490625)
        time.sleep(0.12195125)
        self.tone(987, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1046, 438.109865625)
        time.sleep(0.487805)
        self.tone(880, 218.597615625)
        time.sleep(0.36585375)
        self.tone(783, 108.841490625)
        time.sleep(0.12195125)
        self.tone(880, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1174, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1174, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1046, 218.597615625)
        time.sleep(0.2439025)
        self.tone(987, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1174, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1174, 108.841490625)
        time.sleep(0.12195125)
        self.tone(1046, 108.841490625)
        time.sleep(0.12195125)
        self.tone(987, 218.597615625)
        time.sleep(0.2439025)
        self.tone(1046, 438.109865625)
        time.sleep(0.487805)
        self.tone(880, 438.109865625)
        time.sleep(0.487805)
        self.tone(987, 218.597615625)
        time.sleep(0.2439025)
        self.tone(783, 218.597615625)
        time.sleep(0.2439025)
        self.tone(880, 877.134365625)
        time.sleep(1.2195125)

  

if __name__ == '__main__':
 
    beep = Beep()
    beep.was_wollen()  # Воспроизвести мелодию