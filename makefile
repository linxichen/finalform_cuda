# Paths for Linux CUDA
ICUDA    = /usr/local/cuda-7.0/include
LCUDA    = /usr/local/cuda-7.0/lib64
ICUDA_MAC = /Developer/NVIDIA/CUDA-7.0/include
LCUDA_MAC = /Developer/NVIDIA/CUDA-7.0/lib
ILAPACK = /usr/include/lapacke

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
CXXFLAGS  = -O3 -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(ILAPACK)

# Add CUDA libraries to C++ compiler linking process
LDFLAGS  += -lstdc++ -lcublas -lcurand -lcudart -std=c++11 -L$(LCUDA) -L$(LCUDA_MAC)

# List Executables and Objects
EXEC = vfi

all : $(EXEC)

# Link objects from CUDA and C++ codes
vfi : vfi.o
	$(NVCC) -o $@ $? $(LDFLAGS)

# Compile CUDA code
vfi.o : vfi.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $<

clean :
	rm -f *.o
	rm -f core core.*

veryclean :
	rm -f *.o
	rm -f core core.*
	rm -f $(EXEC)

runvfi : vfi
	./vfi
