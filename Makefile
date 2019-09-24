CXX=g++
CXXFLASG=-std=c++11 -I./
NVCC=nvcc
NVCCFLAGS=$(CXXFLASG) -arch=sm_60

mnist_train_gpu:main_gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<
