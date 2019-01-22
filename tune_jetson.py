import struct
import socket
import cv2
import time

Hn = 0
Sn = 0
Vn = 0
Hx = 0
Sx = 0
Vx = 0

def setHn(x):
    Hn = x
def setSn(x):
    Sn = x
def setVn(x):
    Vn = x
def setHx(x):
    Hx = x
def setSx(x):
    Sx = x
def setVx(x):
    Vx = x
def encode_int_array(int_list):
    buf = struct.pack("!I" + "I" * len(int_list), len(int_list), *int_list)
    return buf
cv2.namedWindow('image')
cv2.createTrackbar('Hn','image',0,180,setHn)
cv2.createTrackbar('Sn','image',0,255,setSn)
cv2.createTrackbar('Vn','image',0,255,setVn)
cv2.createTrackbar('Hx','image',0,180,setHx)
cv2.createTrackbar('Sx','image',0,255,setSx)
cv2.createTrackbar('Vx','image',0,255,setVx)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('0.0.0.0', 88))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            conn.sendall(encode_int_array(
                [Hn, Sn, Vn,Hx, Vx,Sx]))
            cv2.waitKey(25)
