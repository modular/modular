#!/bin/sh
# Reproduces: a module that also uses print()/std.io cannot declare its own
# @extern("write") binding, even with an exactly-matching C signature,
# because @extern treats the C symbol name as a whole-program uniqueness
# key rather than a per-module one, and collides with std.io's own
# internal write() binding used by FileDescriptor.write() / print().
#
# Expected (bug-free) output: both prints succeed.
# Actual output on Mojo 1.0.0b2 (2cf4d08a): fails to lower with
#   "error: existing function with conflicting attributes"
# (see file_descriptor.mojo's own internal write() call site in the trace).
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
