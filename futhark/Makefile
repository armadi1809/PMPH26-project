.DEFAULT_GOAL := compile

FUTHARK := futhark
BACKEND ?= cuda
CFLAGS ?= -O3
DEVICE ?= 0
D ?= 64
N ?= 1024
RUNS ?= 50
THREADS ?=
ENTRIES ?= main64 main128
BENCH_FLAGS ?=

SOURCE := custom-alg1-opt.fut
EXEC := build/custom-alg1-opt-$(BACKEND)

ifeq ($(filter $(BACKEND),cuda multicore c),)
$(error BACKEND must be cuda, multicore, or c)
endif

ifeq ($(BACKEND),cuda)
RUN_ENV := CUDA_VISIBLE_DEVICES="$(DEVICE)"
endif

ifneq ($(strip $(THREADS)),)
ifneq ($(BACKEND),multicore)
$(error THREADS is only supported with BACKEND=multicore)
endif
RUNTIME_FLAGS := --num-threads=$(THREADS)
PASS_FLAGS := --pass-option=--num-threads=$(THREADS)
endif

.PHONY: all check compile check_inputs run bench bench_smoke test test_full clean

all: compile

check:
	$(FUTHARK) check $(SOURCE)
	$(FUTHARK) check smoke.fut

compile: $(EXEC)

$(EXEC): $(SOURCE) Makefile
	@mkdir -p build
	CFLAGS="$(CFLAGS)" $(FUTHARK) $(BACKEND) -o $@ $(SOURCE)

check_inputs:
	@set -eu; \
	case "$(D)" in 64|128) ;; *) printf 'D must be 64 or 128.\n' >&2; exit 1 ;; esac; \
	n="$(N)"; \
	case "$$n" in ''|*[!0-9]*) printf 'N must be a positive multiple of 1024.\n' >&2; exit 1 ;; esac; \
	if [ "$$n" -le 0 ] || [ $$((n % 1024)) -ne 0 ]; then \
	  printf 'N must be a positive multiple of 1024 (online softmax chunk size).\n' >&2; exit 1; \
	fi

run: check_inputs compile
	@set -eu; \
	m=$$(($(N) / $(D))); \
	input="build/input-d$(D)-N$(N).in"; \
	$(FUTHARK) dataset --binary --f32-bounds=1:1 \
	  --generate="[$$m][$(D)][$(D)]f32" \
	  --generate="[$(N)][$(D)]f32" \
	  --generate="[$(N)][$(D)]f32" > "$$input.tmp"; \
	mv "$$input.tmp" "$$input"; \
	$(RUN_ENV) ./$(EXEC) --entry-point=validate $(RUNTIME_FLAGS) < "$$input"

bench:
	@mkdir -p results
	@set -eu; \
	if [ -z "$(strip $(ENTRIES))" ]; then \
	  printf 'ENTRIES must contain at least one benchmark entry.\n' >&2; exit 1; \
	fi; \
	for entry in $(ENTRIES); do \
	  case "$$entry" in main64|main128|validate) ;; \
	    *) printf 'Unsupported benchmark entry: %s\n' "$$entry" >&2; exit 1 ;; \
	  esac; \
	  $(RUN_ENV) CFLAGS="$(CFLAGS)" $(FUTHARK) bench --backend=$(BACKEND) \
	    $(SOURCE) --entry-point="$$entry" --runs=$(RUNS) \
	    $(PASS_FLAGS) $(BENCH_FLAGS) \
	    --json="results/attention-$$entry-$(BACKEND).json"; \
	done

bench_smoke:
	@mkdir -p results
	$(RUN_ENV) CFLAGS="$(CFLAGS)" $(FUTHARK) bench --backend=$(BACKEND) \
	  smoke.fut --runs=$(RUNS) $(PASS_FLAGS) $(BENCH_FLAGS) \
	  --json=results/smoke-$(BACKEND).json

test:
	$(RUN_ENV) CFLAGS="$(CFLAGS)" $(FUTHARK) test --backend=$(BACKEND) \
	  $(PASS_FLAGS) smoke.fut

test_full:
	$(RUN_ENV) CFLAGS="$(CFLAGS)" $(FUTHARK) test --backend=$(BACKEND) \
	  $(PASS_FLAGS) $(SOURCE) --entry-point=validate

clean:
	rm -rf build results data
	rm -f custom-alg1-opt custom-alg1-opt.c custom-alg1-opt.h
	rm -f smoke smoke.c smoke.h
