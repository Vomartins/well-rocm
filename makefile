# Compilers and related
CFLAG         = -Ofast -g
CXX           = hipcc

# DUNE/UMFPACK
DUNE_FLAGS = $(shell pkg-config --cflags --libs dune-istl)
SUITESPARSE_FLAGS = $(shell pkg-config --cflags --libs suitesparse)
UMFPACK_LIBS = -lumfpack -lamd -lsuitesparseconfig

# ROCM/ROCSPARSE
ROCM_PATH = /opt/rocm-6.2.0
ROCSPARSE_INCLUDE = -I$(ROCM_PATH)/include
ROCSPARSE_LIBS = -lrocsparse -lrocblas

LDFLAGS = -Wl,-rpath=$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib $(ROCSPARSE_LIBS) $(DUNE_FLAGS) $(UMFPACK_LIBS)

CFLAGS_ALL = $(CFLAG) $(shell pkg-config --cflags dune-istl) $(shell pkg-config --cflags suitesparse) $(ROCSPARSE_INCLUDE)

# Source code
OBJS= main.o
all:
	@echo "===================================="
	@echo "              Building              "
	@echo "===================================="
	mkdir -p build & mkdir -p build/bin
	rsync -ru main.cpp makefile build/bin
	$(MAKE) -C build/bin WellSolverComp
	cp build/bin/WellSolverComp ./WellSolverComp

WellSolverComp: $(OBJS)
	    $(CXX) $(OBJS) $(LDFLAGS) -o WellSolverComp

%.o : %.cpp
	$(CXX) $(CFLAG_ALL) -c $< -o $@

clean:
	rm -rf ./WellSolverComp build/bin
