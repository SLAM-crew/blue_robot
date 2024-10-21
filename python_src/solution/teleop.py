import cv2
import time
import curses
from motors import set_velocities, stop
from cv import align_histogram, white_balance
from team_light import turn_on_green_light_via_i2c, turn_on_red_light_via_i2c
from manipulator import init_pose

def teleop(stdscr, GREEN=True, RECORD=False, K=0.84):
    if GREEN:
            turn_on_green_light_via_i2c()
    else:
        turn_on_red_light_via_i2c()
    init_pose()
    if RECORD:
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 30, (320, 240))
        cap = cv2.VideoCapture('/dev/video0')
    curses.cbreak()  
    stdscr.keypad(True) 
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.refresh()

    stdscr.addstr(0, 0, "Управляйте роботом с помощью клавиш W, A, S, D. Для выхода нажмите 'q'.")

    speed = 40
    
    while True:
        if RECORD:
            _, frame = cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            y, x, _ = frame.shape
            frame = frame[int(0.5 * y):y, int(0.25 * x):x - int(0.25 * x)]
            frame = align_histogram(frame)
            frame = white_balance(frame)
            out.write(frame)
        key = stdscr.getch()
        
        if key == ord('q'):
            if RECORD:
                out.release()
                cap.release()
            break 
        
        if key == ord('+') and speed < 100:
            speed += 1
            stdscr.addstr(2, 0, f"Текущая скорость: {speed:3d}   ")
        
        elif key == ord('-') and speed > 0:
            speed -= 1 
            stdscr.addstr(2, 0, f"Текущая скорость: {speed:3d}   ")
        
        elif key != -1:
            if key == ord('w'):
                set_velocities(speed, K*speed)        
            elif key == ord('s'):
                set_velocities(-speed, -K*speed)   
            elif key == ord('a'):
                set_velocities(-speed, K*speed)
            elif key == ord('d'):
                set_velocities(speed, -K*speed)
            elif key == ord('x'):
                stop()
        time.sleep(0.1)