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
"""Architecture-specific MMA implementations.

This package contains GPU architecture-specific implementations of matrix
multiply-accumulate (MMA) operations:

- **mma_nvidia**: NVIDIA tensor cores (SM70-SM90) - Volta through Hopper
- **mma_nvidia_sm100**: NVIDIA Blackwell (SM100) tensor cores - 5th gen tensor cores
- **mma_amd**: AMD Matrix Cores (CDNA2/3/4) - Data center GPUs
- **mma_amd_rdna**: AMD WMMA (RDNA3/4) - Consumer GPUs

## Module Organization

Each architecture module contains:
- Private implementation functions (prefixed with `_`)
- Architecture-specific intrinsic calls
- Data type conversions specific to that architecture

## Usage

These modules should **not** be imported directly by user code. Instead, use the
unified interface in `gpu.compute.mma` which automatically dispatches to the
appropriate architecture-specific implementation at compile time:

```mojo
from gpu.compute import mma

# Automatically dispatches to the correct architecture
result = mma(a, b, c)
```

## Internal Implementation Details

The main `gpu.compute.mma` module imports these implementations:

```mojo
from .arch.mma_nvidia import _mma_nvidia
from .arch.mma_amd import _mma_amd
```

And dispatches based on compile-time architecture detection:

```mojo
comptime if is_nvidia_gpu():
    _mma_nvidia(d, a, b, c)
elif is_amd_gpu():
    _mma_amd[block_size](d, a, b, c)
```
"""
