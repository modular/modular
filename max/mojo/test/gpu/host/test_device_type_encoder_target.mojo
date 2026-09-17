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

from std.sys.info import _TargetType, _accelerator_arch, _current_target
from std.testing import assert_true, TestSuite

from std._gpu.host.info import get_gpu_target
from std._plugin._overlay import ADDITIONAL_TARGETS
from max.gpu.host.info import _device_type_encoder_target


def _same_target[a: _TargetType, b: _TargetType]() -> Bool:
    return __mlir_attr[
        `#kgen.param.identical<`, a, `, `, b, `> : !kgen.scalar<bool>`
    ]


def test_encoder_target_follows_the_build() raises:
    comptime target = _device_type_encoder_target()
    comptime if (
        _accelerator_arch() == ""
        or ADDITIONAL_TARGETS.encode_device_types_with_host_layout
    ):
        assert_true(_same_target[target, _current_target()]())
    else:
        assert_true(_same_target[target, get_gpu_target()._mlir_value]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
