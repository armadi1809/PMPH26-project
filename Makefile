.DEFAULT_GOAL := help

.PHONY: help compile run bench test clean

help:
	@printf '%s\n' \
	  'make compile   Compile both GPU baselines (requires CUDA and Futhark).' \
	  'make run       Run both baselines with D=64 N=1024.' \
	  'make bench     Benchmark the four original CFAL datasets.' \
	  'make test      Run small all-ones correctness checks.' \
	  'make clean     Remove generated binaries, inputs, and results.' \
	  'See README.md for individual targets and CPU-only Futhark checks.'

compile run bench test clean:
	$(MAKE) -C cuda $@
	$(MAKE) -C futhark $@
