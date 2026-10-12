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

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from test_utils.cabi_check import (
    AbiConstant,
    AbiTypedef,
    AbiTypedefLike,
    _c_field_decl,
    _c_scalar_name,
    constant_checks,
    emit_cabi_checks_for,
    preamble,
    struct_checks,
    typedef_checks,
)


struct Probe:
    var alpha: Int32
    var beta: UInt8


struct ProbeArray:
    var octets: Array[UInt8, 4]


struct ProbeNested:
    var inner: Probe


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
            "MOJO_CHECK_CABI_FIELD_TYPE(Probe, struct Probe, alpha, int32_t"
            " *ptr)\n"
            "MOJO_CHECK_CABI_FIELD(Probe, struct Probe, beta)"
        ),
    )


def test_struct_checks_array_field() raises:
    assert_equal(
        struct_checks[ProbeArray](),
        (
            "MOJO_CHECK_CABI_LAYOUT(ProbeArray, struct ProbeArray)\n"
            "MOJO_CHECK_CABI_FIELD(ProbeArray, struct ProbeArray, octets)"
        ),
    )


def test_struct_checks_nested_struct_field() raises:
    assert_equal(
        struct_checks[ProbeNested](),
        (
            "MOJO_CHECK_CABI_LAYOUT(ProbeNested, struct ProbeNested)\n"
            "MOJO_CHECK_CABI_FIELD(ProbeNested, struct ProbeNested, inner)\n"
            "MOJO_CHECK_CABI_FIELD_TYPE(ProbeNested, struct ProbeNested,"
            " inner, struct Probe *ptr)"
        ),
    )


def test_c_scalar_name() raises:
    assert_equal(_c_scalar_name[Int8]().value(), "int8_t")
    assert_equal(_c_scalar_name[UInt8]().value(), "uint8_t")
    assert_equal(_c_scalar_name[Int16]().value(), "int16_t")
    assert_equal(_c_scalar_name[UInt16]().value(), "uint16_t")
    assert_equal(_c_scalar_name[Int32]().value(), "int32_t")
    assert_equal(_c_scalar_name[UInt32]().value(), "uint32_t")
    assert_equal(_c_scalar_name[Int64]().value(), "int64_t")
    assert_equal(_c_scalar_name[UInt64]().value(), "uint64_t")


def test_c_scalar_name_not_a_fixed_width_scalar() raises:
    assert_false(_c_scalar_name[String]())
    assert_false(_c_scalar_name[Array[UInt8, 4]]())
    assert_false(_c_scalar_name[SIMD[DType.uint8, 4]]())


def test_c_field_decl_scalar() raises:
    assert_equal(_c_field_decl[Int32]().value(), "int32_t *ptr")
    assert_equal(_c_field_decl[UInt16]().value(), "uint16_t *ptr")


def test_c_field_decl_array() raises:
    assert_equal(
        _c_field_decl[Array[UInt16, 16]]().value(), "uint16_t (*ptr)[16]"
    )


def test_c_field_decl_struct() raises:
    assert_equal(_c_field_decl[Probe]().value(), "struct Probe *ptr")


def test_c_field_decl_skips_bytes() raises:
    assert_false(_c_field_decl[Int8]())
    assert_false(_c_field_decl[UInt8]())
    assert_false(_c_field_decl[Array[Int8, 14]]())
    assert_false(_c_field_decl[Array[UInt8, 4]]())


def test_c_field_decl_array_of_non_scalar() raises:
    with assert_raises(contains="unsupported array type"):
        _ = _c_field_decl[Array[Probe, 2]]()


def test_c_field_decl_simd_vector() raises:
    with assert_raises(contains="unexpected SIMD type"):
        _ = _c_field_decl[SIMD[DType.uint8, 4]]()


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
