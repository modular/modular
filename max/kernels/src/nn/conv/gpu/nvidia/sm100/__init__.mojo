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

"""SM100 Structured Convolution Kernels.

High-performance Conv2D for NVIDIA Blackwell (SM100) GPUs using implicit GEMM
with hardware TMA im2col. Reuses infrastructure from sm100_structured matmul.

Supported: Conv2D fprop with stride=1, dilation=1, BF16/FP16.
"""

from .conv_config import Conv2dConfig, Conv2dProblemShape
from .conv2d import conv2d_fprop, im2col

# Re-export epilogue lambda type for bias/activation fusion
from linalg.utils import elementwise_compute_lambda_type
