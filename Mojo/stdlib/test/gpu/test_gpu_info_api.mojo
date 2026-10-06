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

from std._gpu.host.info import GPUInfo, _all_target_accelerator_values
from std.testing import TestSuite, assert_equal


comptime _MI250X_SPELLINGS = (
    StaticString("gfx90a"),
    StaticString("mi250x"),
    StaticString("amdgpu:gfx90a"),
    StaticString("amd:gfx90a"),
    StaticString("amdgpu:mi250x"),
    StaticString("amd:mi250x"),
)


def test_amd_spellings_resolve() raises:
    """Each accepted spelling of an AMD target must reach the same record.

    Model names are declared as alternative target-accelerator values on their
    corresponding table entries. Vendor prefixes are normalized before that
    lookup, so prefixed model aliases must remain equivalent to their bare
    forms.
    """
    comptime for i in range(len(_MI250X_SPELLINGS)):
        assert_equal(
            GPUInfo.from_name[_MI250X_SPELLINGS[i]]().name,
            "MI250X",
            String("wrong record for ", _MI250X_SPELLINGS[i]),
        )

    assert_equal(GPUInfo.from_name["mi300x"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["amdgpu:mi300x"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["amd:mi300x"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["amdgpu:gfx942"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["amd:gfx942"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["mi355x"]().name, "MI355X")
    assert_equal(GPUInfo.from_name["amdgpu:mi355x"]().name, "MI355X")
    assert_equal(GPUInfo.from_name["amd:mi355x"]().name, "MI355X")

    # MI300A shares the gfx942 ISA with MI300X, so this alias is the only way to
    # reach its distinct record.
    assert_equal(GPUInfo.from_name["amdgpu:mi300a"]().name, "MI300A")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
