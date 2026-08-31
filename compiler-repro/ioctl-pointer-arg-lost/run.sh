#!/bin/sh
# Reproduces: a fixed-arity @extern("ioctl") declaration never correctly
# delivers a pointer out-parameter to the kernel. Sets up its own TCP
# loopback pair (no external oracle needed) and calls
# ioctl(fd, FIONREAD, &avail) after queuing exactly one real byte.
#
# Expected (bug-free) output: "avail: 1".
# Actual output on Mojo 1.0.0b2 (2cf4d08a): non-deterministic across runs
# of this unmodified file -- sometimes "rc: -1 errno: 14" (EFAULT),
# sometimes "rc: 0 ... avail: -99" (claimed success, still unwritten).
# avail never becomes 1 either way. See repro.mojo's header comment for
# the full bisection and the leading theory (Apple arm64's variadic-tail
# stack-passing convention, which a fixed-arity @extern can't express).
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
