#!/bin/sh
# Reproduces: the 1.0.0b1 removal of the legacy __moveinit__/__copyinit__
# spellings is enforced only for the entry file. Imported modules still
# silently accept (and rewrite) them; the entry file rejects them with
# misleading errors.
set -e
cd "$(dirname "$0")"

echo "--- via_import.mojo (imports lib.mojo): compiles+runs today, prints -1 (the acceptance is the bug) ---"
mojo run -I . via_import.mojo

echo
echo "--- main_vs_import.mojo (identical struct inline, run directly): fails today (correct, but with a misleading error) ---"
mojo run main_vs_import.mojo
