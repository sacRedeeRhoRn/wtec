ifneq ($(shell command -v mpicc >/dev/null 2>&1 && echo yes),yes)
MPICC := cc
CPPFLAGS += -DWTEC_NO_MPI
else
MPICC ?= mpicc
endif

OPENMP_PROBE := $(shell tmp=$$(mktemp /tmp/wtec-rgf-omp-XXXXXX 2>/dev/null || echo /tmp/wtec-rgf-omp-test); \
	printf 'int main(void){return 0;}\n' | $(MPICC) -x c - -fopenmp -o $$tmp >/dev/null 2>&1 && echo yes; \
	rm -f $$tmp)
WTEC_RGF_OPENMP ?= auto
ifeq ($(WTEC_RGF_OPENMP),0)
OPENMP_FLAGS :=
else ifeq ($(OPENMP_PROBE),yes)
OPENMP_FLAGS := -fopenmp
else
OPENMP_FLAGS :=
endif

BLAS_BACKEND ?= none
CPPFLAGS += -DWTEC_RGF_BUILD_BLAS_BACKEND=\"$(BLAS_BACKEND)\"
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Iinclude $(CPPFLAGS) $(OPENMP_FLAGS)
LDFLAGS ?= -lm $(OPENMP_FLAGS)

SRC := $(wildcard src/*.c)
OBJ := $(patsubst src/%.c,build/%.o,$(SRC))
BIN := build/wtec_rgf_runner

.PHONY: all clean

all: $(BIN)

build:
	mkdir -p build

build/%.o: src/%.c | build
	$(MPICC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf build
