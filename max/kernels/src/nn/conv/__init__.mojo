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
"""Convolution kernels (1D, 2D, 3D, transposed).

Shared utilities live here (conv_utils), with platform-specific
implementations organized by vendor and architecture:

  conv/                       - shared utilities and main conv implementations
  conv/conv.mojo              - CPU direct conv, GPU conv via cuDNN/MIOpen
  conv/conv_transpose.mojo    - transposed (deconvolution) for CPU and GPU
  conv/conv_utils.mojo        - ConvShape, ConvPartition, epilogue helpers
  conv/gpu/nvidia/sm100/      - SM100 (Blackwell) structured Conv2D kernels
"""
