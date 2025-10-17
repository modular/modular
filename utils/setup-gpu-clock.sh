#!/bin/bash
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

if ! [ $(id -u) = 0 ]; then
   echo "The script should be run as root." >&2
   exit 1
fi

# Configures Nvidia GPU settings. These settings should persist
# across reboots because we're setting `persistence-mode=1`.
set_nvidia_gpu_config() {
  echo "set_nvidia_gpu_config: Setting NVidia GPU config".

  nvidia-smi --auto-boost-permission=0
  nvidia-smi -pm 1
  nvidia-smi --auto-boost-default=0
  nvidia-smi --auto-boost-permission=1
  nvidia-smi --persistence-mode=1
  nvidia-smi -acp 0
  for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
    nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i
  done
}


# Configures AMD GPUs. These settings are important for performance testing.
set_amd_gpu_config() {
	echo "set_amd_gpu_config: Setting AMD GPU config".

	rocm-smi --setperfdeterminism 1900
}


# Auto-detect GPU type
if command -v nvidia-smi >/dev/null 2>&1; then
  set_nvidia_gpu_config
elif command -v rocm-smi >/dev/null 2>&1; then
  set_amd_gpu_config
else
  echo "No NVIDIA or AMD GPU attached"
fi