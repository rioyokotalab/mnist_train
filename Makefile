CXX=g++
CXXFLASG=-std=c++11 -I./
NVCC=nvcc
NVCCFLAGS=$(CXXFLASG) -arch=sm_60

TARGET=cpu

build_target:mnist_train_$(TARGET)

mnist_train_cpu:main_cpu.cpp
	$(CXX) $(CXXFLASG) -o $@ $<

mnist_train_gpu:main_cpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<
