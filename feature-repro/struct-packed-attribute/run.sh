#!/bin/sh
# TDD red test for a feature request: no below-natural alignment
# override exists. `@packed` fails to parse, and the existing `@align(N)`
# is documented (and verified) as a minimum that silently clamps
# below-natural requests up to the natural alignment.
#
# Current (red) result on Mojo 1.0.0b2 (2cf4d08a) and 1.0.0 (ed45d567):
#   error: use of unknown declaration 'packed'
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
