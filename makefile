# Compilers and related
CFLAG         = -Ofast -g
CXX           = g++

# DUNE
LDFLAGS_DUNE = -ldunecommon # -ldunegeometry
DUNE_FLAGS = $(shell pkg-config --cflags --libs dune-istl)
SUITESPARSE_FLAGS = $(shell pkg-config --cflags --libs suitesparse)
UMFPACK_LIBS = -lumfpack -lamd -lsuitesparseconfig

LDFLAGS = $(DUNE_FLAGS) $(UMFPACK_LIBS)

CFLAGS_ALL = $(CFLAG) $(shell pkg-config --cflags dune-istl) $(shell pkg-config --cflags suitesparse)

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
