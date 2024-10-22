import time
import math
import requests

raspberry_ip = '192.168.2.12' # 192.168.2.53


url = f'http://{raspberry_ip}:5000'
traj = [[100, 100], [100, 150], [50, 150], [50, 100], [100, 100]]

def trajectory_send(trajectory, flag):
    data = {"trajectory": trajectory, 'flag': flag}
    start_time = time.time()
    response = requests.post(url + '/trajectory_points', json=data)
    if response.status_code == 200:
        print(f"Response from Raspberry Pi: {response.json()['message']}")
        end_time = time.time() - start_time
        print(end_time)
    else:
        print(f"Failed to send string. Status code: {response.status_code}, Error: {response.text}")

def recover_angle(axis, angle):
    data = {'axis': axis, 'angle': angle}
    start_time = time.time()
    response = requests.post(url + '/recover_angle', json=data)
    if response.status_code == 200:
        print(f"Response from Raspberry Pi: {response.json()['message']}")
        end_time = time.time() - start_time
        print(end_time)
    else:
        print(f"Failed to send string. Status code: {response.status_code}, Error: {response.text}")

# trajectory_send([[94, 120], [35, 120], [35, 140], [10, 140]], 'cube')
# recover_angle('N', -math.pi / 6)