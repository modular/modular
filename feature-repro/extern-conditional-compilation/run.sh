#!/bin/sh
# TDD red test for a feature request: today this file fails to parse.
# It should go GREEN (parse, compile, run, and print the final line)
# once either of these ships:
#   - a comptime-computed string literal is accepted by @extern, or
#     module-level `comptime if` becomes legal so the symbol choice can
#     be made via a real conditional instead (FACT 1, see repro.mojo);
#   - a single C symbol can be declared under more than one arity in one
#     module (FACT 2, see repro.mojo).
#
# Current (red) result on Mojo 1.0.0b2 (2cf4d08a):
#   error: '@extern' requires a string literal argument
# (comment out the ERRNO_SYM declaration in repro.mojo to instead see
# FACT 2 in isolation: "error: duplicate functions named 'fcntl'".
# Uncommenting the FACT-1a block at the top of the file, in a scratch
# copy, on its own, shows a third error: "'comptime if' must be contained
# in a function".)
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
