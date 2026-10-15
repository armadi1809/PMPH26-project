.DEFAULT_GOAL := help

SIZES ?=

.PHONY: help compile run bench test clean datasets verify

help:
	@printf '%s\n' \
	  'make compile   Compile both GPU baselines (requires CUDA and Futhark).' \
	  'make run       Run both baselines with D=64 N=1024.' \
	  'make bench     Benchmark the four original CFAL datasets.' \
	  'make test      Run small all-ones correctness checks.' \
	  'make datasets  Generate random inputs and Futhark reference outputs in data/.' \
	  'make verify    Check the CUDA baseline against the reference outputs.' \
	  'make clean     Remove generated binaries, inputs, and results.' \
	  'Restrict datasets/verify with SIZES, e.g. SIZES="64,1024 128,1024".' \
	  'See README.md for individual targets and CPU-only Futhark checks.'

compile run bench test clean:
	$(MAKE) -C cuda $@
	$(MAKE) -C futhark $@

datasets:
	./gen-datasets.sh $(SIZES)

verify:
	./verify.sh $(SIZES)
