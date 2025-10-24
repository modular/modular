# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #
"""GPU compute operations package - MMA and tensor core operations.

This package provides GPU tensor core and matrix multiplication operations:

- **mma**: Warp matrix-multiply-accumulate (WMMA) operations for SM70-SM90
- **mma_sm100**: SM100/Blackwell-specific MMA operations
- **mma_util**: Utility functions for MMA operations
- **mma_operand_descriptor**: Operand descriptor types for MMA
- **tensor_ops**: Tensor core-based reductions and operations
- **tcgen05**: 5th generation tensor core operations (Blackwell)

These modules should typically be imported directly. For example:

```mojo
from gpu.compute import mma
result = mma.mma[M, N, K](a, b, c)
```
"""
