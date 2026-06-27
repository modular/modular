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
"""Provides the Apple silicon GPU backend implementations for matmuls."""

from .matmul_8x8 import gemm_kernel_apple_8x8
from .matmul_kernel import (
    AppleM5MatMul,
    enqueue_apple_conv2d,
    enqueue_apple_matmul,
    enqueue_apple_matmul_split_k,
)
from linalg.arch.apple.mma import ConvIm2colParams
