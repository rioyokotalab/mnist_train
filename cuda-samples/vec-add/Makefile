NVCC=nvcc
NVCCFLAGS=-std=c++11 -arch=sm_60
TARGET=vec_add.test

$(TARGET):main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
