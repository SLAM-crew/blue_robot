import socket

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the server IP and port
server_ip = '127.0.0.1'  # Replace with the actual IP address of the server
port = 2001

# Connect to the server
client_socket.connect((server_ip, port))

# Send data (you can simulate I2C data here)
message = "Hello from Raspberry Pi"
client_socket.send(message.encode('utf-8'))

# Receive acknowledgment from the server
data = client_socket.recv(1024).decode('utf-8')
print(f"Received from server: {data}")

# Close the connection
client_socket.close()