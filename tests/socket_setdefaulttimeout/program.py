import socket

socket.setdefaulttimeout(None) # OK
socket.setdefaulttimeout(0) # OK
socket.setdefaulttimeout(2.4) # OK

socket.setdefaulttimeout(-3) # DyLin warn
socket.setdefaulttimeout(-3.4) # DyLin warn