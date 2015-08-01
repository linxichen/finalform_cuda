# Quick tutorial
# $@ = target
# $^ = all depdencies
# $? = dependcies more recent than target
# $< = only the first dependency

# Paths for includes I, and libraries L
ICUDA     = /usr/local/cuda-7.0/include
LCUDA     = /usr/local/cuda-7.0/lib64
ICUDA_MAC = /Developer/NVIDIA/CUDA-7.0/include
LCUDA_MAC = /Developer/NVIDIA/CUDA-7.0/lib
ICPP_MAC  = /usr/local/include
LCPP_MAC  = /usr/local/lib
ILAPACK   = /usr/include/lapacke

SDIR     = .
IDIR     = .
LDIR     = .

# Compiler for CUDA
NVCC      = nvcc

# CUDA compiling options
NVCCFLAGS =  -arch sm_30 #-use_fast_math

# Compiler for C code
CXX       = g++

# Standard optimization flags to C++ compiler
CXXFLAGS  = -O2 -std=c++11 -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(ILAPACK)

# Add CUDA libraries to C++ compiler linking process
LDFLAGS  += -lcublas -lcurand -lcudart -larmadillo -lopenblas -llapacke -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC)

# List Executables and Objects
EXEC = vfi

all : $(EXEC)

# Link objects from CUDA and C++ codes
EXEC : vfi.o vfi_link.o cppcode.o
	$(NVCC) -o $@ $^ $(LDFLAGS)

# Prepare CUDA objects for linking
vfi_link.o : vfi.o cppcode.o
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -dlink $^ -o $@

# Compile CUDA code
vfi.o : vfi.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $^

# Compile C++ code
cppcode.o : cppcode.cpp
	$(CXX) $(CXXFLAGS) -c -std=c++11  $^

clean :
	rm -f *.o
	rm -f core core.*

veryclean :
	rm -f *.o
	rm -f core core.*
	rm -f $(EXEC)

runvfi : vfi
	./vfi
