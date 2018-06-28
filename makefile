# Quick tutorial
# $@ = target
# $^ = all depdencies
# $? = dependcies more recent than target
# $< = only the first dependency

# Paths for includes I, and libraries L
ICUDA     = /usr/local/cuda-7.0/include
ICUDA_MAC = /Developer/NVIDIA/CUDA-7.0/include
ICPP_MAC  = /usr/local/include
LCUDA     = /usr/local/cuda-7.0/lib64
LCUDA_MAC = /Developer/NVIDIA/CUDA-7.0/lib
LCPP_MAC  = /usr/lib

SDIR = .
IDIR = ./cudatools/include
LDIR = ./cudatools/lib

INCL+= -I$(ICUDA)
INCL+= -I$(ICUDA_MAC)
INCL+= -I$(ICPP_MAC)
INCL+= -I$(IDIR)

# Compiler for CUDA
NVCC = nvcc

# CUDA compiling options
NVCCFLAGS  = -O3 -lineinfo -g -std=c++11

# Compiler for C code
CXX = g++

# Standard optimization flags to C++ compiler
CXXFLAGS = -O3 -Wall -Wextra -pedantic-errors -std=c++11

# Add CUDA libraries to C++ compiler linking process
LDLIBS += -lcublas
LDLIBS += -lcurand
LDLIBS += -lcudart -lcudadevrt
LDLIBS += -larmadillo
# LDLIBS += -larmadillo -lopenblas -llapack -larpack
LDLIBS += -lcudatools
LDLIBS += -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC) -L$(LDIR)

# List Executables and Objects
EXEC = ks

all : veryclean $(EXEC)

# Link objects from CUDA and C++ codes
$(EXEC) : ks_dlink.o ks.o $(LDIR)/libcudatools.a
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDLIBS)

#Prepare for host linker
ks_dlink.o : ks.o
	$(NVCC) -dlink $^ -o $@ $(NVCCFLAGS) $(LDLIBS)

# Compile CUDA code
ks.o : ks.cu
	$(NVCC) $(NVCCFLAGS) $(INCL) -dc $^ -o $@ $(LDLIBS)

clean :
	rm -f *.o
	rm -f core core.*

veryclean :
	rm -f *.o
	rm -f core core.*
	rm -f $(EXEC)

run : ks
	./ks
