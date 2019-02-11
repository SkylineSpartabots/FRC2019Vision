import time
import cv2


# Create capture
cap = cv2.VideoCapture(0)

# Check if cap is open
if cap.isOpened() is not True:
    quit()
# Create videowriter as a SHM sink
#out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)
cap.set(cv2.CAP_PROP_AUTOFOCUS,0);
# Loop it
i = 0
while True:
    # Get the frame
    ret, frame = cap.read()
    # Check
    if ret is True:
        cv2.imwrite("C:\\Users\\NeilHazra\\chessboard\\" + str(i) +".jpg", frame)
        cv2.imshow("video", frame);
        cv2.waitKey(100)
        i = i+1
cap.release()
