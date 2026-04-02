#!/bin/bash

set -euo pipefail

if [[ $OSTYPE == darwin* ]]; then
  platform=mac
elif [[ $(uname -m) == "x86_64" ]]; then
  platform=intel
else
  platform=graviton
fi

root="$PWD/external/+http_archive+llvm-ifs/tools/$platform"

if [[ "${MACOS:-}" == "true" ]]; then
  exec "$root/llvm-readtapi.stripped" -arch arm64 -extract "$1" -o "$2"
else
  exec "$root/llvm-ifs.stripped" "$1" --output-elf="$2"
fi
