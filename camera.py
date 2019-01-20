import time
import cv2

# Cam properties
fps = 1
frame_width = 640
frame_height = 480
# Create capture
cap = cv2.VideoCapture(1)
# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0);
# Check if cap is open
if cap.isOpened() is not True:
    quit()
# Create videowriter as a SHM sink
#out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)

# Loop it
i = 0
while True:
    # Get the frame
    ret, frame = cap.read()
    # Check
    if ret is True:
        # Flip frame
        frame = cv2.flip(frame, 1)
        cv2.imwrite("C:\\Users\\NeilHazra\\chessboard\\" + str(i) +".jpg", frame)
        cv2.imshow("video", frame);
        cv2.waitKey(100)
        i = i+1
cap.release()
