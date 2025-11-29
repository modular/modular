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
"""MMA operand descriptor trait (deprecated - use `gpu.compute.mma_operand_descriptor`).

This module is deprecated. For new code, import from the `gpu.compute.mma_operand_descriptor`
module instead:

```mojo
# Deprecated:
from gpu.mma_operand_descriptor import MMAOperandDescriptor

# Recommended:
from gpu.compute.mma_operand_descriptor import MMAOperandDescriptor
```

This module provides the trait for MMA operand descriptors used in matrix
multiply-accumulate operations.
"""

# Re-export all public symbols from compute.mma_operand_descriptor for backward compatibility
from .compute.mma_operand_descriptor import MMAOperandDescriptor
