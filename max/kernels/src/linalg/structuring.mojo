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

"""Re-exports the shared memory pointer and barrier types for GPU kernels."""


# Shared memory pointer and barrier type declarations.
# Canonical definitions live in structured_kernels.smem_types; re-exported
# here for backward compatibility so existing
# `from linalg.structuring import ...` statements continue to work.
from structured_kernels.smem_types import (
    SMemBarrier,
    SMemArray,
    SMemPtr,
)
