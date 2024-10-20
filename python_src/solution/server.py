from flask import Flask, request, jsonify
from motors import set_velocities, stop
import time
app = Flask(__name__)
def move_forward():
    set_velocities(40, 40)

def turn_left():
    set_velocities(-40, 40)

def turn_right():
    set_velocities(40, -40)

def stop_movement():
    stop()
def move_back():
    set_velocities(-40, -40)


def handle_received_string(received_string):
    commands = {
        'FORWARD' :  move_forward,
        'LEFT' :  turn_left,
        'BACK' :  move_back,
        'RIGHT' :  turn_right,
        'STOP' :  stop_movement
    }
    received_string= received_string.strip().upper() 
    if received_string in commands:
        commands[received_string]()
    else:
        print('unknown command')


@app.route('/receive_string', methods=['POST'])
def receive_string():

    try:
        start_time = time.time()

        data = request.json
        received_string = data.get('string')

        if received_string:
            response_message = handle_received_string(received_string)
          
            return jsonify({"status": "success", "message": response_message}), 200
        else:
            return jsonify({"status": "error", "message": "No string received"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='192.168.2.53', port=5000)

