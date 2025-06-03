import socket

socket.setdefaulttimeout(None) # OK
socket.setdefaulttimeout(0) # OK
socket.setdefaulttimeout(2.4) # OK

try: socket.setdefaulttimeout(-3) # DyLin warn
except ValueError as e: pass

try: socket.setdefaulttimeout(-3.4) # DyLin warn
except ValueError as e: pass