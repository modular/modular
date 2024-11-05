##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#!/bin/bash

echo $kplot
shopt -s expand_aliases
dir="$(pwd)"
source $MODULAR_PATH/utils/start-modular.sh
cd $dir

kplot output_base.csv output_branch.csv -o base_vs_branch
kplot output_base.csv output_branch.csv -o base_vs_branch -x pdf
kplot output_base.csv output_branch.csv -o base_vs_branch_label \
	--label="baseline" --label="experiment"
kplot output_base.csv output_branch.csv -o base_vs_branch_label \
	--label="baseline" --label="experiment" -x pdf
kplot output_base.csv output_branch.csv -o base_vs_branch_label_compare \
	--label="baseline" --label="experiment" -c
kplot output_base.csv output_branch.csv output_cublas.csv -o base_vs_branch_vs_cublas \
	--label="baseline" --label="experiment" --label="cublas" -f
kplot output_base.csv output_branch.csv output_cublas.csv -o base_vs_branch_vs_cublas \
	--label="baseline" --label="experiment" --label="cublas" -f -x pdf
