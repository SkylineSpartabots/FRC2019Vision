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
min_area = 1;
min_solidity = 0.6;
aspectRatio = 1;
tolerance = 1;

def setHn(x):
    global Hn
    Hn = x
def setSn(x):
    global Sn
    Sn = x
def setVn(x):
    global Vn
    Vn = x
def setHx(x):
    global Hx
    Hx = x
def setSx(x):
    global Sx
    Sx = x
def setVx(x):
    global Vx
    Vx = x

def setMinArea(x):
    global min_area
    min_area = x
def setMinSolidity(x):
    global min_solidity;
    min_solidity = x/100.0
def setAspectRatio(x):
    global aspectRatio;
    aspectRatio = x/100.0
def aspectRatioTolerance(x):
    global tolerance;
    tolerance = x/100.0;
cv2.namedWindow('image')
cv2.createTrackbar('Hn','image',0,180,setHn)
cv2.createTrackbar('Sn','image',0,255,setSn)
cv2.createTrackbar('Vn','image',0,255,setVn)
cv2.createTrackbar('Hx','image',0,180,setHx)
cv2.createTrackbar('Sx','image',0,255,setSx)
cv2.createTrackbar('Vx','image',0,255,setVx)

cv2.createTrackbar('minArea','image',0,1000,setMinArea)
cv2.createTrackbar('minSolidity','image',0,100,setMinSolidity)
cv2.createTrackbar('aspectRatio','image',0,600,setAspectRatio)
cv2.createTrackbar('rationTolerance','image',0,400,aspectRatioTolerance)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(('10.29.76.86',5807))
    print ("connected");
    while True:
        string = "%" + str(Hn) + ";" + str(Sn) + ";" + str(Vn) + ";" + str(Hx) + ";" + str(Sx) + ";" + str(Vx) + ";" + str(min_area) + ";" + str(min_solidity) + ";" + str(aspectRatio) + ";" + str(tolerance) + "%";
        print(string);
        s.sendall(bytes(string,'utf-8'));
        cv2.waitKey(200)
