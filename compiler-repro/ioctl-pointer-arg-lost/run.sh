#!/bin/sh
# Reproduces: a fixed-arity @extern("ioctl") declaration of the variadic
# ioctl() never delivers the pointer out-parameter on arm64 macOS; the
# lowering passes all declared parameters via the fixed-argument
# convention while the callee reads its variadic tail from the stack.
#
# Expected (bug-free) output: "... avail: 1".
# Actual on Mojo 1.0.0b2 (2cf4d08a) AND 1.0.0 (ed45d567): "avail: -99",
# the cell is never written; rc/errno vary by program context but are
# stable within one binary. external_call["ioctl", Int32,
# num_fixed_args=2] on 1.0.0 is the working alternative.
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
