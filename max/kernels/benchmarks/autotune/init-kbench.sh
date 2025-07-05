#!/bin/bash

kbench_path="$(realpath $(dirname $BASH_SOURCE))"
pip install -r $kbench_path/requirements.txt
alias kbench="python3 $kbench_path/kbench.py"
export KERNEL_BENCHMARKS_ROOT="$(realpath $kbench_path/..)"
