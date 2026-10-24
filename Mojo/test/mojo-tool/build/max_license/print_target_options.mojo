# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Test the --print-supported-accelerators target information option.

# Test --print-supported-accelerators lists supported GPU architectures
# RUN: %mojo-build --print-supported-accelerators 2>&1 | FileCheck %s --check-prefix=CHECK_ACCELERATORS

# Test error when --print-supported-accelerators combined with other print options
# RUN: not %mojo-build --print-supported-accelerators --print-supported-targets 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR_MULTI

# CHECK_ERROR_MULTI: error: only one --print-* option can be specified at a time

# Sections print in target-name order: amdgpu, metal, nvptx.
# CHECK_ACCELERATORS: Supported Accelerator Architectures:
# CHECK_ACCELERATORS: AMD (ROCm/HIP):
# CHECK_ACCELERATORS: gfx942
# CHECK_ACCELERATORS: gfx950
# CHECK_ACCELERATORS: Apple Silicon GPU:
# CHECK_ACCELERATORS: apple-m4
# CHECK_ACCELERATORS: NVIDIA (CUDA):
# CHECK_ACCELERATORS: sm_80
# CHECK_ACCELERATORS: sm_90


def main():
    pass
