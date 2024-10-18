import time
import xr_gpio as gpio


class Beep(object):
    def __init__(self):
        self.buzzer_pin = gpio.BUZZER  # Подключение к буззеру
       

        self.notes = {
            'A4': 440,
            'C5': 523,
            'D5': 587,
            'E5': 659,
            'F5': 698,
            'G4': 392,
            'G5': 784,
            'B4': 494,
            'REST': 0
        }

          # Полная мелодия "Was wollen wir trinken" с новым продолжением
        self.melody = [
            # Первая часть
            'A4', 'C5', 'D5', 'E5', 'E5', 'F5', 'D5', 'E5', 'REST',
            'D5', 'D5', 'C5', 'B4', 'C5', 'A4', 'REST', 'A4',
            'D5', 'D5', 'C5', 'B4',
            'D5', 'D5', 'C5', 'B4',
            'C5', 'A4', 'REST', 'G4', 'A4',
            # Вторая часть (новая)
            'A4', 'C5', 'D5', 'G5', 'G5', 'F5', 'E5', 'D5', 'REST',
            'D5', 'E5', 'F5', 'G5', 'F5', 'E5', 'D5', 'REST',
            'E5', 'F5', 'G5', 'E5',
            'D5', 'C5', 'B4', 'A4', 'REST',
            # Третья часть (продолжение с изменёнными нотами)
            'A4', 'C5', 'D5', 'G5', 'G5', 'F5', 'E5', 'D5', 'REST',
            'C5', 'D5', 'E5', 'F5', 'E5', 'D5', 'C5', 'REST',
            'B4', 'C5', 'D5', 'C5',
            'B4', 'A4', 'REST', 'G4', 'A4'
        ]

        # Длительность каждой ноты в мелодии
        self.durations = [
            # Первая часть
            4, 4, 4, 3, 3, 4, 4, 2, 4,
            4, 6, 6, 6, 2, 3, 8, 4,
            4, 4, 4, 4,
            4, 6, 6, 6,
            2, 3, 8, 2, 1,
            # Вторая часть
            4, 4, 4, 3, 3, 4, 4, 2, 4,
            4, 6, 6, 6, 2, 3, 8, 4,
            4, 4, 4, 4,
            4, 6, 6, 6,
            2, 3, 8, 2, 1,
            # Третья часть
            4, 4, 4, 3, 3, 4, 4, 2, 4,
            4, 6, 6, 6, 2, 3, 8, 4,
            4, 4, 4, 4,
            4, 6, 6, 6,
            2, 3, 8, 2, 1,
        ]

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

    def play_music(self):
        for i in range(len(self.melody)):
            note = self.notes[self.melody[i]]
            duration = 1000 / self.durations[i]
            self.tone(note, duration)
            pause_between_notes = duration * 0.30
            time.sleep(pause_between_notes / 1000)

if __name__ == '__main__':
    beep = Beep()
    beep.play_music()