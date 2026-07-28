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

"""The MAX library for hardware-accelerated programming in Mojo.

The kernels in this library provide the building blocks for AI
inference and other compute-intensive workloads on CPU and GPU, including
[linear algebra](/api/mojo/linalg/) functions like matrix multiplication,
[neural network operators](/api/mojo/nn/) such as attention and convolution,
[quantization](/api/mojo/quantization/) routines, [key-value
caches](/api/mojo/kv_cache/) for transformer models, and primitives for
[multi-GPU communication](/api/mojo/comm/) and [extending a MAX
graph](/api/mojo/extensibility/) with custom operations.
"""
