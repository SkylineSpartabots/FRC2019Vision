SHELL := /bin/sh
LIB_PATH=/home/ubuntu/ntcore/
NT_INCLUDE_PATH=/home/ubuntu/ntcore/include/
WPI_INCLUDE_PATH=/home/ubuntu/ntcore/wpiutil/include/
FILENAME=jetson_image_processing.cu
OUTNAME=vision_client
all: clean	
	nvcc --std=c++11 -L$(LIB_PATH) -I$(NT_INCLUDE_PATH) -I$(WPI_INCLUDE_PATH) $(FILENAME) -lntcore -lwpiutil -lpthread -o $(OUTNAME) `pkg-config --cflags --libs opencv` 
 
clean:
	#-rm ${OUTNAME}
 
run: 
	./startProc.sh
	./${OUTNAME}

