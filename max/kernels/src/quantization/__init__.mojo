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
"""This package contains a set of APIs for quantizing tensor data.

Quantization is a technique used to reduce the precision of floating-point
numbers, which are used in most neural networks. Quantization is a type of
lossy compression, which means that some precision is lost, but the resulting
tensors take less memory and computations are faster.
"""

from .per_channel_grouped_4bit import *
