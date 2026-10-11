.DEFAULT_GOAL := compile

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -diag-suppress=1650
ARCH ?=
DEVICE ?= 0
D ?= 64
N ?= 1024
RUNS ?= 50
SIZES ?= 64,16384 64,32768 128,8192 128,16384

ifneq ($(strip $(ARCH)),)
NVCCFLAGS += -arch=$(ARCH)
endif

EXEC := flash_attention_gpu
SRC := flash_attention_gpu.cu

.PHONY: all compile run bench test clean

all: compile

compile: $(EXEC)

$(EXEC): $(SRC) Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC)

run: compile
	CUDA_VISIBLE_DEVICES="$(DEVICE)" ./$(EXEC) $(D) $(N) $(RUNS)

test: compile
	CUDA_VISIBLE_DEVICES="$(DEVICE)" ./$(EXEC) 64 1024 1
	CUDA_VISIBLE_DEVICES="$(DEVICE)" ./$(EXEC) 128 1024 1

bench: compile
	@mkdir -p results
	@set -eu; \
	if [ -z "$(strip $(SIZES))" ]; then \
	  printf 'SIZES must contain at least one d,N pair.\n' >&2; exit 1; \
	fi; \
	for size in $(SIZES); do \
	  d=$${size%,*}; n=$${size#*,}; \
	  output="results/d$$d-N$$n.runtimes"; \
	  printf '\nBenchmarking d=%s N=%s (%s measured runs)\n' "$$d" "$$n" "$(RUNS)"; \
	  if ! CUDA_VISIBLE_DEVICES="$(DEVICE)" ./$(EXEC) "$$d" "$$n" "$(RUNS)" > "$$output.tmp"; then \
	    rm -f "$$output.tmp"; \
	    exit 1; \
	  fi; \
	  mv "$$output.tmp" "$$output"; \
	  cat "$$output"; \
	done

clean:
	rm -f $(EXEC)
	rm -rf results
