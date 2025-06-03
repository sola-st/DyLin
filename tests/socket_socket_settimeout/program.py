import socket

# create a new socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.settimeout(None) # OK
s.settimeout(0) # OK
s.settimeout(2.4) # OK

try: s.settimeout(-3) # DyLin warn
except ValueError as e: pass

try: s.settimeout(-3.4) # DyLin warn
except ValueError as e: pass