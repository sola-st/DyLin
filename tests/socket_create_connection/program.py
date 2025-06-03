import socket
import socketserver

# Define the host and port to connect to
host = 'www.google.com'
port = 80

socket.create_connection((host, port), None) # OK
socket.create_connection((host, port), 0) # OK
socket.create_connection((host, port), 2.4) # OK

socket.create_connection((host, port), -3) # DyLin warn
socket.create_connection((host, port), -3.4) # DyLin warn