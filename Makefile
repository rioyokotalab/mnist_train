CXX=g++
CXXFLASG=-std=c++11 -I./
NVCC=nvcc
NVCCFLAGS=$(CXXFLASG) -arch=sm_60
TARGET=mnist_train_gpu

$(TARGET):main_gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
