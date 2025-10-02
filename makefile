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

# EIGEN
EIGEN_INC = -I/usr/include/eigen3

LDFLAGS = -Wl,-rpath=$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib $(ROCSPARSE_LIBS) $(DUNE_FLAGS) $(UMFPACK_LIBS)

CFLAGS_ALL = $(CFLAG) $(shell pkg-config --cflags dune-istl) $(shell pkg-config --cflags suitesparse) $(ROCSPARSE_INCLUDE) $(EIGEN_INC)

# Source code
OBJS= main.o \
    linearSystemData.o \
    umfpackSolver.o \
    conversionData.o
all:
	@echo "===================================="
	@echo "              Building              "
	@echo "===================================="
	mkdir -p build & mkdir -p build/bin
	rsync -ru main.cpp \
	          linearSystemData.cpp linearSystemData.hpp \
		      umfpackSolver.cpp umfpackSolver.hpp \
			  conversionData.cpp conversionData.hpp \
			  makefile build/bin
	$(MAKE) -C build/bin WellSolverComp
	cp build/bin/WellSolverComp ./WellSolverComp

WellSolverComp: $(OBJS)
	    $(CXX) $(OBJS) $(LDFLAGS) -o WellSolverComp

%.o : %.cpp
	$(CXX) $(CFLAGS_ALL) -c $< -o $@

clean:
	rm -rf ./WellSolverComp build/bin
