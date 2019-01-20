nvcc jetson_image_processing.cu -o compiled.o `pkg-config --cflags --libs opencv` --std=c++11
./compiled.o