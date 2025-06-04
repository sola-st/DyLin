import socket
import socketserver

# Define the host and port to connect to
host = 'www.google.com'
port = 80

try: socket.create_connection((host, port), None) # OK
except Exception as e: pass

try: socket.create_connection((host, port), 0) # OK
except Exception as e: pass

try: socket.create_connection((host, port), 2.4) # OK
except Exception as e: pass

try: socket.create_connection((host, port), -3) # DyLin warn
except Exception as e: pass

try: socket.create_connection((host, port), -3.4) # DyLin warn
except Exception as e: pass