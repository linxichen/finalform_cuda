# Quick tutorial
# $@ = target
# $^ = all depdencies
# $? = dependcies more recent than target
# $< = only the first dependency

# Paths for includes I, and libraries L
ICUDA = /usr/local/cuda-7.0/include
LCUDA = /usr/local/cuda-7.0/lib64
ICUDA_MAC = /Developer/NVIDIA/CUDA-7.0/include
LCUDA_MAC = /Developer/NVIDIA/CUDA-7.0/lib
ICPP_MAC = /usr/local/include
LCPP_MAC = /usr/local/lib

SDIR = .
IDIR = ./cudatools/include
LDIR = ./cudatools/lib

# Compiler for CUDA
NVCC = nvcc

# CUDA compiling options
NVCCFLAGS = -lineinfo -g -arch sm_30 #-use_fast_math

# Compiler for C code
CXX = g++

# Standard optimization flags to C++ compiler
CXXFLAGS = -O2 -std=c++11 -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(IDIR)

# Add CUDA libraries to C++ compiler linking process
LDLIBS += -lstdc++ -lcublas -lcurand -lcudart -lcudadevrt -larmadillo -lopenblas -llapack -larmadillo -lcudatools -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC) -L$(LDIR)

# List Executables and Objects
EXEC = vfi

all : veryclean $(EXEC)

# Link objects from CUDA and C++ codes
$(EXEC) : vfi_dlink.o vfi.o $(LDIR)/libcudatools.a
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDLIBS)

#Prepare for host linker
vfi_dlink.o : vfi.o
	$(NVCC) -dlink  $^ -o $@ $(NVCCFLAGS) $(LDLIBS)

# Compile CUDA code
vfi.o : vfi.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS)  -dc $^ -o $@ $(LDLIBS)

clean :
	rm -f *.o
	rm -f core core.*

veryclean :
	rm -f *.o
	rm -f core core.*
	rm -f $(EXEC)

runvfi : vfi
	./vfi
