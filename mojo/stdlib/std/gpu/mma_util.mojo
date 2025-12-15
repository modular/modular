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
"""Matrix multiply utilities (deprecated - use `gpu.compute.mma_util`).

This module is deprecated. For new code, import from the `gpu.compute.mma_util`
module instead:

```mojo
# Deprecated:
from gpu.mma_util import load_matrix_a, load_matrix_b, store_matrix_d

# Recommended:
from gpu.compute.mma_util import load_matrix_a, load_matrix_b, store_matrix_d
```

This module provides utility functions for loading and storing matrices for
matrix multiply-accumulate operations.
"""

# Re-export all public symbols from compute.mma_util for backward compatibility
from .compute.mma_util import (
    load_matrix_a,
    load_matrix_a_amd,
    load_matrix_b,
    load_matrix_b_amd,
    store_matrix_d,
)
