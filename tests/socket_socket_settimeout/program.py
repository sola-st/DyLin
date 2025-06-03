import socket

# create a new socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.settimeout(None) # OK
s.settimeout(0) # OK
s.settimeout(2.4) # OK

s.settimeout(-3) # DyLin warn
s.settimeout(-3.4) # DyLin warn