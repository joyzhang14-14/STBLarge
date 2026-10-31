NVCC ?= nvcc
ARCH ?= sm_121
FLAGS = -O2 -std=c++17 -arch=$(ARCH) -Xcompiler -fPIC

libstbl.so: kernels.cu model.cu gguf.cpp kernels.h gguf.h stbl.h
	$(NVCC) $(FLAGS) -shared kernels.cu model.cu gguf.cpp -o $@

clean:
	rm -f libstbl.so
