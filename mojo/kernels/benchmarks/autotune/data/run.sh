##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
