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

"""Model test bed: abstract harness + runner for layer benchmarking, profiling,
IR dumping, and correctness testing."""

from testbed.harness import CompiledLayerBundle, LayerTestHarness
from testbed.registry import HARNESS_REGISTRY, register_harness
from testbed.runner import LayerTestRunner

__all__ = [
    "HARNESS_REGISTRY",
    "CompiledLayerBundle",
    "LayerTestHarness",
    "LayerTestRunner",
    "register_harness",
]
