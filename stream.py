import time
import cv2

# Cam properties
fps = 30.
frame_width = 640
frame_height = 480
# Create capture
cap = cv2.VideoCapture("/dev/video2")

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)

# Define the gstreamer sink
gst_str_rtp = "appsrc ! videoconvert ! video/x-raw, format=(string)I420, width=(int)640, height=(int)480 ! omxh264enc bitrate=600000 ! video/x-h264, stream-format=(string)byte-stream ! h264parse ! rtph264pay ! udpsink host=10.29.76.21 port=5000 sync=true "

# Check if cap is open
if cap.isOpened() is not True:
    print "Cannot open camera. Exiting."
    quit()

# Create videowriter as a SHM sink
out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)

# Loop it
while True:
    # Get the frame
    ret, frame = cap.read()
    # Check
    if ret is True:
        # Flip frame
        frame = cv2.flip(frame, 1)
        # Write to SHM
	#cv2.imshow("video", frame);        
	cv2.waitKey(1)	
	out.write(frame)
cap.release()
