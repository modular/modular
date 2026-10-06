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

from std.testing import TestSuite, assert_equal, assert_true
from test_utils.cabi_check import (
    AbiConstant,
    AbiTypedef,
    AbiTypedefLike,
    constant_checks,
    emit_cabi_checks_for,
    preamble,
    struct_checks,
    typedef_checks,
)


struct Probe:
    var alpha: Int32
    var beta: UInt8


def test_preamble() raises:
    var out = preamble(["<errno.h>", "<netinet/in.h>"])
    assert_true(out.startswith("// Generated ABI reference; do not edit."))
    assert_true("#include <errno.h>\n" in out)
    assert_true("#include <netinet/in.h>\n" in out)
    assert_true("#include <stddef.h>\n" in out)
    assert_true("#include <stdint.h>\n" in out)
    assert_true("#define MOJO_CHECK_CABI_LAYOUT(name, type)" in out)
    assert_true("#define MOJO_CHECK_CABI_FIELD(name, type, field)" in out)
    assert_true("#define MOJO_CHECK_CABI_CONST(name)" in out)
    assert_true(
        "#define MOJO_CHECK_CABI_IS_SIGNED(e) ((__typeof__(e))-1 < 0)" in out
    )


def test_struct_checks() raises:
    assert_equal(
        struct_checks[Probe](),
        (
            "MOJO_CHECK_CABI_LAYOUT(Probe, struct Probe)\n"
            "MOJO_CHECK_CABI_FIELD(Probe, struct Probe, alpha)\n"
            "MOJO_CHECK_CABI_FIELD(Probe, struct Probe, beta)"
        ),
    )


def test_typedef_checks() raises:
    assert_equal(
        typedef_checks[AbiTypedef["probe_t", UInt16]](),
        "MOJO_CHECK_CABI_LAYOUT(probe_t, probe_t)",
    )


def test_constant_checks() raises:
    assert_equal(constant_checks("EPERM"), "MOJO_CHECK_CABI_CONST(EPERM)")


def test_emit_cabi_checks_for() raises:
    comptime Typedef = AbiTypedef["probe_t", UInt16]
    assert_equal(
        emit_cabi_checks_for(
            includes=["<errno.h>"],
            structs=TypeList.of[Trait=AnyType, Probe](),
            typedefs=TypeList.of[Trait=AbiTypedefLike, Typedef](),
            constants=[AbiConstant("EPERM", Int32(1))],
        ),
        preamble(["<errno.h>"])
        + "\n\n"
        + struct_checks[Probe]()
        + "\n\n"
        + typedef_checks[Typedef]()
        + "\n\n"
        + constant_checks("EPERM"),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
