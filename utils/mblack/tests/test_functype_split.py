# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests for splitting long function-type expressions (e.g. FFI signatures like
# `def() thin abi("C") -> ...`). A `named_effect`'s parens stay inline; a long
# signature wraps at the parameter or return-type bracket instead.

from tests.util import assert_mojo_format


def test_ffi_functype_thin_abi_is_stable():
    """A thin C-ABI FFI signature formats unchanged, with `abi("C")` inline."""
    source = (
        "from std.ffi import OwnedDLHandle, c_char\n"
        "\n"
        "\n"
        "def main() raises:\n"
        '    var lib = OwnedDLHandle("libcurl.dylib")\n'
        "    var result = String(capacity=256)\n"
        "    var curl_version = lib.get_function[\n"
        '        def() thin abi("C") -> UnsafePointer[\n'
        "            c_char, ImmutOrigin(origin_of(result))\n"
        "        ]\n"
        '    ]("curl_version")\n'
        "    _ = curl_version\n"
        "    _ = lib\n"
    )
    assert_mojo_format(source, source)


def test_long_functype_splits_return_type_not_named_effect():
    """A too-long functype splits at the return-type bracket, `abi("C")` intact.

    The named effect's parenthesized string must stay on one line rather than
    being chosen as the split point.
    """
    source = (
        "from std.ffi import OwnedDLHandle, c_char\n"
        "\n"
        "\n"
        "def main() raises:\n"
        '    var lib = OwnedDLHandle("libcurl.dylib")\n'
        "    var result_version_string_buffer = String(capacity=256)\n"
        "    var curl_version = lib.get_function[\n"
        '        def() thin abi("C") -> UnsafePointer[c_char,'
        " ImmutOrigin(origin_of(result_version_string_buffer))]\n"
        '    ]("curl_version")\n'
        "    _ = curl_version\n"
        "    _ = lib\n"
    )
    expected = (
        "from std.ffi import OwnedDLHandle, c_char\n"
        "\n"
        "\n"
        "def main() raises:\n"
        '    var lib = OwnedDLHandle("libcurl.dylib")\n'
        "    var result_version_string_buffer = String(capacity=256)\n"
        "    var curl_version = lib.get_function[\n"
        '        def() thin abi("C") -> UnsafePointer[\n'
        "            c_char, ImmutOrigin(origin_of(result_version_string_buffer))\n"
        "        ]\n"
        '    ]("curl_version")\n'
        "    _ = curl_version\n"
        "    _ = lib\n"
    )
    assert_mojo_format(source, expected)


def test_async_ffi_functype_is_stable():
    """An async function type also keeps its named effect inline."""
    source = (
        "from std.ffi import OwnedDLHandle, c_char\n"
        "\n"
        "\n"
        "def main() raises:\n"
        '    var lib = OwnedDLHandle("libcurl.dylib")\n'
        "    var result_version_string_buffer = String(capacity=256)\n"
        "    var curl_version = lib.get_function[\n"
        '        async def() thin abi("C") -> UnsafePointer[\n'
        "            c_char, ImmutOrigin(origin_of(result_version_string_buffer))\n"
        "        ]\n"
        '    ]("curl_version")\n'
        "    _ = curl_version\n"
        "    _ = lib\n"
    )
    assert_mojo_format(source, source)


def test_unsplittable_functype_left_intact():
    """A too-long functype whose only bracket is the effect is left intact.

    With empty params and a non-bracketed return type there is no other bracket
    to wrap at, so the line is emitted unchanged.
    """
    source = (
        "from std.ffi import OwnedDLHandle\n"
        "\n"
        "\n"
        "comptime AVeryLongReturnTypeAliasNameToForceThisSignatureOverColumnLimit"
        " = Float64\n"
        "\n"
        "\n"
        "def main() raises:\n"
        '    var lib = OwnedDLHandle("libm.dylib")\n'
        "    var f = lib.get_function[\n"
        '        def() thin abi("C") ->'
        " AVeryLongReturnTypeAliasNameToForceThisSignatureOverColumnLimit\n"
        '    ]("sym")\n'
        "    _ = f\n"
        "    _ = lib\n"
    )
    assert_mojo_format(source, source)


def test_functype_with_params_splits_at_parameter_list():
    """A functype with a plain (non-bracketed) return type splits at its params.

    The parameter list, not the `abi("C")` effect, is the split point, so the
    C-ABI signature stays readable (an FFI-heavy file like `_cpython.mojo`
    relies on this).
    """
    source = (
        "from std.ffi import OwnedDLHandle\n"
        "\n"
        "\n"
        "def main() raises:\n"
        '    var lib = OwnedDLHandle("libm.dylib")\n'
        "    var some_math_function = lib.get_function[\n"
        "        def(Float64, Float64, Float64, Float64, Float64, Float64)"
        ' thin abi("C") -> Float64\n'
        '    ]("some_symbol_name")\n'
        "    _ = some_math_function\n"
        "    _ = lib\n"
    )
    expected = (
        "from std.ffi import OwnedDLHandle\n"
        "\n"
        "\n"
        "def main() raises:\n"
        '    var lib = OwnedDLHandle("libm.dylib")\n'
        "    var some_math_function = lib.get_function[\n"
        "        def(\n"
        "            Float64, Float64, Float64, Float64, Float64, Float64\n"
        '        ) thin abi("C") -> Float64\n'
        '    ]("some_symbol_name")\n'
        "    _ = some_math_function\n"
        "    _ = lib\n"
    )
    assert_mojo_format(source, expected)


def test_real_def_still_splits_at_parameters():
    """Regression guard: a genuine `def` with no effect splits at its params."""
    source = (
        "def a_function_with_a_fairly_long_name("
        "argument_one: Int, argument_two: Int, arg3: Int) -> Int:\n"
        "    return argument_one\n"
    )
    expected = (
        "def a_function_with_a_fairly_long_name(\n"
        "    argument_one: Int, argument_two: Int, arg3: Int\n"
        ") -> Int:\n"
        "    return argument_one\n"
    )
    assert_mojo_format(source, expected)


def test_real_def_with_named_effect_and_empty_params():
    """A statement-level `def` with a named effect keeps the effect inline too.

    `named_effect` appears on both function types and real definitions, so a
    genuine `def foo() abi("C") -> ...:` header wraps at the return type.
    """
    source = (
        "def some_ffi_wrapper_function_with_a_long_name()"
        ' abi("C") -> SIMD[DType.float64, 4]:\n'
        "    return SIMD[DType.float64, 4](0)\n"
    )
    expected = (
        "def some_ffi_wrapper_function_with_a_long_name() abi(\"C\") -> (\n"
        "    SIMD[DType.float64, 4]\n"
        "):\n"
        "    return SIMD[DType.float64, 4](0)\n"
    )
    assert_mojo_format(source, expected)
