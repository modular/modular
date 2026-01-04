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
"""SM100 Structured Matmul - Refactored with encapsulated pipeline management.

This module provides the same SM100 matmul functionality as the original sm100
module, but with improved code organization:

Key abstractions:
- WorkIterator/SchedulerWorkIterator: Encapsulate work iteration and pipeline state
- TilePipeline/OutputTilePipeline: Encapsulate producer-consumer synchronization
- TileLoaderTMA: Encapsulate TMA tile loading logic
- Context managers for cleaner acquire/release patterns

## Switching Implementations

### Option 1: Environment Variable (Recommended)

Set `MODULAR_USE_STRUCTURED_SM100=1` to use this implementation:

```bash
# Use original sm100 (default):
./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test

# Use sm100_structured:
MODULAR_USE_STRUCTURED_SM100=1 ./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test
```

### Option 2: Direct Import

```mojo
# Original:
from linalg.matmul.gpu.sm100.matmul import (
    blackwell_matmul_tma_umma_warp_specialized
)

# Structured (this module):
from linalg.matmul.gpu.sm100_structured import (
    blackwell_matmul_tma_umma_warp_specialized
)
```

See DOCS/testing_and_switching.md for full documentation.
"""

from .matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
    matmul_sm100_fallback,
)
