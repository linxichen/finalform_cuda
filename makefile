# Paths for Linux CUDA
ICUDA    = /usr/local/cuda-6.5/include
LCUDA    = /usr/local/cuda-6.5/lib64
ICUDA_MAC = /Developer/NVIDIA/CUDA-6.5/include
LCUDA_MAC = /Developer/NVIDIA/CUDA-6.5/lib
ICPP_MAC = /usr/local/include
LCPP_MAC = /usr/local/lib
ILAPACK = /usr/include/lapacke

SDIR     = .
IDIR     = .
LDIR     = .

# Compiler for CUDA
NVCC      = /Developer/NVIDIA/CUDA-6.5/bin/nvcc

# CUDA compiling options
NVCCFLAGS =  -arch sm_30 #-use_fast_math

# Compiler for C code
CXX       = g++

# Standard optimization flags to C++ compiler
CXXFLAGS  = -O2 -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(ILAPACK)

# Add CUDA libraries to C++ compiler linking process
LDFLAGS  += -lcublas -lcurand -lcudart -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC)

# List Executables and Objects
EXEC = vfi 

all : $(EXEC)

# Link objects from CUDA and C++ codes
vfi : vfi.o
	$(CXX) -o $@ $? $(LDFLAGS)

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
