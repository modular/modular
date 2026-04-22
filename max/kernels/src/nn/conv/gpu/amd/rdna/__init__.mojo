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

"""RDNA Conv2D via implicit GEMM (fused im2col + WMMA matmul).

High-performance Conv2D for AMD RDNA 3+ GPUs. Fuses im2col coordinate
computation into the WMMA matmul kernel's A-tile loader, eliminating the
large intermediate im2col buffer.

Supported: Conv2D fprop with stride=1, dilation=1, BF16/FP16.
"""

from .dispatch import dispatch_rdna_conv2d
