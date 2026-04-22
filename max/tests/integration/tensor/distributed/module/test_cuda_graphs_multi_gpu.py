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
"""CUDA graph tests on real GPU -- inherits all logic from _cuda_graphs."""

from max.driver import Accelerator

from module._cuda_graphs import CUDAGraphTests


class TestCUDAGraphs(CUDAGraphTests):
    DEVICE = Accelerator(0)
