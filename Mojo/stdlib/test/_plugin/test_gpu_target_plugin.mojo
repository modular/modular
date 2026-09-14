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

from std._plugin._overlay import STD_PLUGINS
from std.testing import assert_equal, TestSuite

from max.gpu.host.info import (
    _a100_target,
    _h100_target,
    _metal_m1_target,
    _mi300x_target,
)


# STD_PLUGINS order: [DefaultPlugin=0, MetalPlugin=1, CUDAPlugin=2, HIPPlugin=3]


def _idx[target: __mlir_type.`!kgen.target`]() -> Int:
    return Int(SIMDLength(mlir_value=STD_PLUGINS.index_for_target[target]))


def test_nvidia_targets_select_cuda_plugin() raises:
    assert_equal(_idx[_a100_target._mlir_value](), 2)
    assert_equal(_idx[_h100_target._mlir_value](), 2)


def test_amd_targets_select_hip_plugin() raises:
    assert_equal(_idx[_mi300x_target._mlir_value](), 3)


def test_apple_targets_select_metal_plugin() raises:
    assert_equal(_idx[_metal_m1_target._mlir_value](), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
