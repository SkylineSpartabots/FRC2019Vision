import struct
import socket
import cv2
import time


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(('10.29.76.86',5803))
    while True:
        data = s.recv(1024);
        print(repr(data))
