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
IDIR = .
LDIR = .

# Compiler for CUDA
NVCC = nvcc

# CUDA compiling options
NVCCFLAGS = -arch sm_30 #-use_fast_math

# Compiler for C code
CXX = g++

# Standard optimization flags to C++ compiler
CXXFLAGS = -O2 -std=c++11 -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(ILAPACK)

# Add CUDA libraries to C++ compiler linking process
LDLIBS += -lstdc++ -lcublas -lcurand -lcudart -larmadillo -lopenblas -llapack -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC)

# List Executables and Objects
EXEC = vfi

all : veryclean $(EXEC) runvfi

# Link objects from CUDA and C++ codes
$(EXEC) : vfi.o vfi_dlink.o cppcode.o
	$(NVCC) $^ $(LDLIBS) -o $@

# Dlink CUDA relocatable object into executable object
vfi_dlink.o : vfi.o
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(LDLIBS) -dlink $^ -o $@

# Compile CUDA code
vfi.o : vfi.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(LDLIBS) -dc $^ -o $@

# Compile C++ code
cppcode.o : cppcode.cpp
	$(CXX) $(CXXFLAGS) $(LDLIBS) -c $^ -o $@

clean :
	rm -f *.o
	rm -f core core.*

veryclean :
	rm -f *.o
	rm -f core core.*
	rm -f $(EXEC)

runvfi : vfi
	./vfi
