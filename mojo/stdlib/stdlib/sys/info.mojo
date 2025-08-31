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
"""Implements methods for querying the host target info.

You can import these APIs from the `sys` package. For example:

```mojo
from sys import CompilationTarget

print(CompilationTarget.is_x86())
```
"""

from collections.string.string_slice import _get_kgen_string

from .ffi import _external_call_const, external_call

alias _TargetType = __mlir_type.`!kgen.target`


@always_inline("nodebug")
fn _current_target() -> _TargetType:
    return __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`


@register_passable("trivial")
struct CompilationTarget[value: _TargetType = _current_target()]:
    """A struct that provides information about a target architecture.

    This struct encapsulates various methods to query target-specific
    information such as architecture features, OS details, endianness, and
    memory characteristics.

    Parameters:
        value: The target architecture to query. Defaults to the current target.
    """

    @always_inline("nodebug")
    @staticmethod
    fn unsupported_target_error[
        result: AnyType = NoneType._mlir_type,
        *,
        operation: Optional[String] = None,
        note: Optional[String] = None,
    ]() -> result:
        """Produces a constraint failure when called indicating that some
        operation is not supported by the current compilation target.

        Parameters:
            result: The never-returned result type of this function.
            operation: Optional name of the operation that is not supported.
                Should be a function name or short description.
            note: Optional additional note to print.

        Returns:
            This function does not return normally, however a return type
            can be specified to satisfy Mojo type checking.
        """

        alias note_text = " Note: " + note.value() if note else ""

        @parameter
        if operation:
            constrained[
                False,
                "Current compilation target does not support operation: "
                + operation.value()
                + "."
                + note_text,
            ]()
        else:
            constrained[
                False,
                "Current compilation target does not support this operation."
                + note_text,
            ]()

        return os.abort[result]()

    @always_inline("nodebug")
    @staticmethod
    fn _has_feature[name: StaticString]() -> Bool:
        """Checks if the target has a specific feature.

        Parameters:
            name: The name of the feature to check.

        Returns:
            True if the target has the specified feature, False otherwise.
        """
        return __mlir_attr[
            `#kgen.param.expr<target_has_feature,`,
            Self.value,
            `,`,
            _get_kgen_string[name](),
            `> : i1`,
        ]

    @always_inline("nodebug")
    @staticmethod
    fn _arch() -> StaticString:
        return StaticString(Self.__arch())

    @always_inline("nodebug")
    @staticmethod
    fn __arch() -> __mlir_type.`!kgen.string`:
        return __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            Self.value,
            `, "arch" : !kgen.string`,
            `> : !kgen.string`,
        ]

    @staticmethod
    fn _is_arch[name: StaticString]() -> Bool:
        """Helper function to check if the target architecture is the same as
        given by the name.

        NOTE: This function is needed so that we don't compare the strings at
        compile time using `==`, which would lead to a recursions due to SIMD
        (and potentially many other things) depending on architecture checks.

        Parameters:
            name: The name to check against the target architecture.

        Returns:
            True if the target architecture is the same as the given name,
            False otherwise.
        """
        return __mlir_attr[
            `#kgen.param.expr<eq,`,
            Self.__arch(),
            `, `,
            _get_kgen_string[name](),
            `> : i1`,
        ]

    @always_inline("nodebug")
    @staticmethod
    fn _os() -> StaticString:
        var res = __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            Self.value,
            `, "os" : !kgen.string`,
            `> : !kgen.string`,
        ]
        return StaticString(res)

    @always_inline("nodebug")
    @staticmethod
    fn default_compile_options() -> StaticString:
        """Returns the default compile options for the compilation target.

        Returns:
            The string of default compile options for the compilation target.
        """

        @parameter
        if is_triple["nvptx64-nvidia-cuda", Self.value]():
            # TODO: use `is_nvidia_gpu` when moved to into this struct.
            return "nvptx-short-ptr=true"
        else:
            return ""

    # Features

    @staticmethod
    fn has_sse4() -> Bool:
        """Checks if the target supports SSE4 instructions.

        Returns:
            True if the target supports SSE4, False otherwise.
        """
        return Self._has_feature["sse4"]()

    @staticmethod
    fn has_avx() -> Bool:
        """Returns True if the host system has AVX, otherwise returns False.

        Returns:
            True if the host system has AVX, otherwise returns False.
        """
        return Self._has_feature["avx"]()

    @staticmethod
    fn has_avx2() -> Bool:
        """Returns True if the host system has AVX2, otherwise returns False.

        Returns:
            True if the host system has AVX2, otherwise returns False.
        """
        return Self._has_feature["avx2"]()

    @staticmethod
    fn has_avx512f() -> Bool:
        """Returns True if the host system has AVX512, otherwise returns False.

        Returns:
            True if the host system has AVX512, otherwise returns False.
        """
        return Self._has_feature["avx512f"]()

    @staticmethod
    fn has_intel_amx() -> Bool:
        """Returns True if the host system has Intel AMX support, otherwise returns
        False.

        Returns:
            True if the host system has Intel AMX and False otherwise.
        """
        return Self._has_feature["amx-tile"]()

    @staticmethod
    fn has_fma() -> Bool:
        """Returns True if the target has FMA (Fused Multiply-Add) support,
        otherwise returns False.

        Returns:
            True if the target has FMA support, otherwise returns False.
        """
        return Self._has_feature["fma"]()

    @staticmethod
    fn has_vnni() -> Bool:
        """Returns True if the target has avx512_vnni, otherwise returns False.

        Returns:
            True if the target has avx512_vnni, otherwise returns False.
        """
        return (
            Self._has_feature["avx512vnni"]() or Self._has_feature["avxvnni"]()
        )

    @staticmethod
    fn has_neon() -> Bool:
        """Returns True if the target has Neon support, otherwise returns
        False.

        Returns:
            True if the target support the Neon instruction set.
        """
        return Self._has_feature["neon"]() or Self.is_apple_silicon()

    @staticmethod
    fn has_neon_int8_dotprod() -> Bool:
        """Returns True if the target has the Neon int8 dot product extension,
        otherwise returns False.

        Returns:
            True if the target support the Neon int8 dot product extension and
            False otherwise.
        """
        return Self.has_neon() and Self._has_feature["dotprod"]()

    @staticmethod
    fn has_neon_int8_matmul() -> Bool:
        """Returns True if the target has the Neon int8 matrix multiplication
        extension (I8MM), otherwise returns False.

        Returns:
            True if the target support the Neon int8 matrix multiplication
            extension (I8MM) and False otherwise.
        """
        return Self.has_neon() and Self._has_feature["i8mm"]()

    # Platforms

    @staticmethod
    fn is_x86() -> Bool:
        """Checks if the target is an x86 architecture.

        Returns:
            True if the target is x86, False otherwise.
        """
        return Self.has_sse4()

    @staticmethod
    fn is_apple_m1() -> Bool:
        """Check if the target is an Apple M1 system.

        Returns:
            True if the host system is an Apple M1, False otherwise.
        """
        return Self._is_arch["apple-m1"]()

    @staticmethod
    fn is_apple_m2() -> Bool:
        """Check if the target is an Apple M2 system.

        Returns:
            True if the host system is an Apple M2, False otherwise.
        """
        return Self._is_arch["apple-m2"]()

    @staticmethod
    fn is_apple_m3() -> Bool:
        """Check if the target is an Apple M3 system.

        Returns:
            True if the host system is an Apple M3, False otherwise.
        """
        return Self._is_arch["apple-m3"]()

    @staticmethod
    fn is_apple_m4() -> Bool:
        """Check if the target is an Apple M4 system.

        Returns:
            True if the host system is an Apple M4, False otherwise.
        """
        return Self._is_arch["apple-m4"]()

    @staticmethod
    fn is_apple_silicon() -> Bool:
        """Check if the host system is an Apple Silicon with AMX support.

        Returns:
            True if the host system is an Apple Silicon with AMX support, and
            False otherwise.
        """
        return (
            Self.is_apple_m1()
            or Self.is_apple_m2()
            or Self.is_apple_m3()
            or Self.is_apple_m4()
        )

    @staticmethod
    fn is_neoverse_n1() -> Bool:
        """Returns True if the host system is a Neoverse N1 system, otherwise
        returns False.

        Returns:
            True if the host system is a Neoverse N1 system and False otherwise.
        """
        return Self._is_arch["neoverse-n1"]()

    # OS

    @staticmethod
    fn is_linux() -> Bool:
        """Returns True if the host operating system is Linux.

        Returns:
            True if the host operating system is Linux and False otherwise.
        """
        return Self._os() == "linux"

    @staticmethod
    fn is_macos() -> Bool:
        """Returns True if the host operating system is macOS.

        Returns:
            True if the host operating system is macOS and False otherwise.
        """
        return Self._os() in ["darwin", "macosx"]

    @staticmethod
    fn is_windows() -> Bool:
        """Returns True if the host operating system is Windows.

        Returns:
            True if the host operating system is Windows and False otherwise.
        """
        return Self._os() == "windows"


fn platform_map[
    T: Copyable & Movable, //,
    operation: Optional[String] = None,
    *,
    linux: Optional[T] = None,
    macos: Optional[T] = None,
    windows: Optional[T] = None,
]() -> T:
    """Helper for defining a compile time value depending
    on the current compilation target, raising a compilation
    error if trying to access the value on an unsupported target.

    Example:

    ```mojo
    alias EDEADLK = platform_alias["EDEADLK", linux=35, macos=11]()
    ```

    Parameters:
        T: The type of the value.
        operation: The operation to show in the compilation error.
        linux: Optional support for linux targets.
        macos: Optional support for macos targets.
        windows: Optional support for windows targets.
    """

    @parameter
    if CompilationTarget.is_macos() and macos:
        return macos.value()
    elif CompilationTarget.is_linux() and linux:
        return linux.value()
    elif CompilationTarget.is_windows() and windows:
        return windows.value()
    else:
        return CompilationTarget.unsupported_target_error[
            T, operation=operation
        ]()


@always_inline("nodebug")
fn _accelerator_arch() -> StaticString:
    """Returns the accelerator architecture string for the current target
    accelerator.

    If there is no accelerator on the system, this function returns an empty
    string.

    Returns:
        The accelerator architecture string for the current target accelerator.
    """
    return StaticString(
        __mlir_attr.`#kgen.param.expr<accelerator_arch> : !kgen.string`
    )


@always_inline("nodebug")
fn _triple_attr[
    target: _TargetType = _current_target()
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        target,
        `, "triple" : !kgen.string`,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
fn is_triple[
    name: StringLiteral, target: _TargetType = _current_target()
]() -> Bool:
    """Returns True if the target triple of the compiler matches the input and
    False otherwise.

    Parameters:
      name: The name of the triple value.
      target: The triple value to be checked against.

    Returns:
        True if the triple matches and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _triple_attr[target](),
        `, `,
        name.value,
        `> : i1`,
    ]


@always_inline("nodebug")
fn _is_sm_8x() -> Bool:
    return (
        is_nvidia_gpu["sm_80"]()
        or is_nvidia_gpu["sm_86"]()
        or is_nvidia_gpu["sm_89"]()
    )


@always_inline("nodebug")
fn _is_sm_9x() -> Bool:
    return is_nvidia_gpu["sm_90"]() or is_nvidia_gpu["sm_90a"]()


@always_inline("nodebug")
fn _is_sm_100x() -> Bool:
    return is_nvidia_gpu["sm_100"]() or is_nvidia_gpu["sm_100a"]()


@always_inline("nodebug")
fn _is_sm_101x() -> Bool:
    return is_nvidia_gpu["sm_101"]() or is_nvidia_gpu["sm_101a"]()


@always_inline("nodebug")
fn _is_sm_120x() -> Bool:
    return is_nvidia_gpu["sm_120"]() or is_nvidia_gpu["sm_120a"]()


@always_inline("nodebug")
fn _has_blackwell_tcgen05() -> Bool:
    return is_nvidia_gpu["sm_100a"]() or is_nvidia_gpu["sm_101a"]()


@always_inline("nodebug")
fn _is_sm_8x_or_newer() -> Bool:
    return _is_sm_8x() or _is_sm_9x_or_newer()


@always_inline("nodebug")
fn _is_sm_9x_or_newer() -> Bool:
    return _is_sm_9x() or _is_sm_100x_or_newer()


@always_inline("nodebug")
fn _is_sm_100x_or_newer() -> Bool:
    return _is_sm_100x() or _is_sm_120x_or_newer()


@always_inline("nodebug")
fn _is_sm_120x_or_newer() -> Bool:
    return _is_sm_120x()


@always_inline("nodebug")
fn is_apple_gpu() -> Bool:
    """Returns True if the target triple is for Apple GPU (Metal) and False otherwise.
    Returns:
        True if the triple target is Apple GPU and False otherwise.
    """
    return is_triple["air64-apple-macosx"]()


@always_inline("nodebug")
fn is_apple_gpu[subarch: StaticString]() -> Bool:
    """Returns True if the target triple of the compiler is `air64-apple-macosx`
    and we are compiling for the specified sub-architecture and False otherwise.

    Parameters:
        subarch: The subarchitecture (e.g. sm_80).

    Returns:
        True if the triple target is cuda and False otherwise.
    """
    return is_apple_gpu() and CompilationTarget._is_arch[subarch]()


@always_inline("nodebug")
fn is_nvidia_gpu() -> Bool:
    """Returns True if the target triple of the compiler is `nvptx64-nvidia-cuda`
    False otherwise.

    Returns:
        True if the triple target is cuda and False otherwise.
    """
    return is_triple["nvptx64-nvidia-cuda"]()


@always_inline("nodebug")
fn is_nvidia_gpu[subarch: StaticString]() -> Bool:
    """Returns True if the target triple of the compiler is `nvptx64-nvidia-cuda`
    and we are compiling for the specified sub-architecture and False otherwise.

    Parameters:
        subarch: The subarchitecture (e.g. sm_80).

    Returns:
        True if the triple target is cuda and False otherwise.
    """
    return is_nvidia_gpu() and CompilationTarget._is_arch[subarch]()


@always_inline("nodebug")
fn _is_amd_rdna3() -> Bool:
    return (
        is_amd_gpu["amdgpu:gfx1100"]()
        or is_amd_gpu["amdgpu:gfx1101"]()
        or is_amd_gpu["amdgpu:gfx1102"]()
        or is_amd_gpu["amdgpu:gfx1103"]()
        # These last two are technically RDNA3.5, but we'll treat them as RDNA3
        # for now.
        or is_amd_gpu["amdgpu:gfx1150"]()
        or is_amd_gpu["amdgpu:gfx1151"]()
    )


@always_inline("nodebug")
fn _is_amd_rdna4() -> Bool:
    return is_amd_gpu["amdgpu:gfx1200"]() or is_amd_gpu["amdgpu:gfx1201"]()


@always_inline("nodebug")
fn _is_amd_rdna() -> Bool:
    return _is_amd_rdna3() or _is_amd_rdna4()


@always_inline("nodebug")
fn _is_amd_mi300x() -> Bool:
    return is_amd_gpu["amdgpu:gfx942"]()


@always_inline("nodebug")
fn _is_amd_mi355x() -> Bool:
    return is_amd_gpu["amdgpu:gfx950"]()


@always_inline("nodebug")
fn _cdna_version() -> Int:
    constrained[
        _is_amd_mi300x() or _is_amd_mi355x(),
        "querying the cdna version is only supported on AMD hardware",
    ]()

    @parameter
    if _is_amd_mi300x():
        return 3
    else:
        return 4


@always_inline("nodebug")
fn _cdna_3_or_newer() -> Bool:
    @parameter
    if is_amd_gpu():
        return _cdna_version() >= 3
    return False


@always_inline("nodebug")
fn _cdna_4_or_newer() -> Bool:
    @parameter
    if is_amd_gpu():
        return _cdna_version() >= 4
    return False


@always_inline("nodebug")
fn _is_amd_cdna() -> Bool:
    return _is_amd_mi300x() or _is_amd_mi355x()


@always_inline("nodebug")
fn is_amd_gpu() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    False otherwise.

    Returns:
        True if the triple target is amdgpu and False otherwise.
    """
    return is_triple["amdgcn-amd-amdhsa"]()


@always_inline("nodebug")
fn is_amd_gpu[subarch: StaticString]() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    and we are compiling for the specified sub-architecture, False otherwise.

    Returns:
        True if the triple target is amdgpu and False otherwise.
    """
    return is_amd_gpu() and _accelerator_arch() == subarch


@always_inline("nodebug")
fn is_gpu() -> Bool:
    """Returns True if the target triple is GPU and False otherwise.

    Returns:
        True if the triple target is GPU and False otherwise.
    """
    return is_nvidia_gpu() or is_amd_gpu()


@always_inline("nodebug")
fn is_little_endian[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the target's endianness is little and False otherwise.

    Parameters:
        target: The target architecture.

    Returns:
        True if the target is little endian and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "endianness" : !kgen.string`,
            `> : !kgen.string`,
        ],
        `,`,
        `"little" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn is_big_endian[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the target's endianness is big and False otherwise.

    Parameters:
        target: The target architecture.

    Returns:
        True if the target is big endian and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "endianness" : !kgen.string`,
            `> : !kgen.string`,
        ],
        `,`,
        `"big" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn is_32bit[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the maximum integral value is 32 bit.

    Parameters:
        target: The target architecture.

    Returns:
        True if the maximum integral value is 32 bit, False otherwise.
    """
    return size_of[DType.index, target]() == size_of[DType.int32, target]()


@always_inline("nodebug")
fn is_64bit[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the maximum integral value is 64 bit.

    Parameters:
        target: The target architecture.

    Returns:
        True if the maximum integral value is 64 bit, False otherwise.
    """
    return size_of[DType.index, target]() == size_of[DType.int64, target]()


@always_inline("nodebug")
fn simd_bit_width[target: _TargetType = _current_target()]() -> Int:
    """Returns the vector size (in bits) of the specified target.

    Parameters:
        target: The target architecture.

    Returns:
        The vector size (in bits) of the specified target.
    """
    return Int(
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "simd_bit_width" : !kgen.string`,
            `> : index`,
        ]
    )


@deprecated("Use `sys.simd_bit_width()` instead.")
@always_inline("nodebug")
fn simdbitwidth[target: _TargetType = _current_target()]() -> Int:
    return simd_bit_width[target]()


@always_inline("nodebug")
fn simd_byte_width[target: _TargetType = _current_target()]() -> Int:
    """Returns the vector size (in bytes) of the specified target.

    Parameters:
        target: The target architecture.

    Returns:
        The vector size (in bytes) of the host system.
    """
    alias CHAR_BIT = 8
    return simd_bit_width[target]() // CHAR_BIT


@deprecated("Use `sys.simd_byte_width()` instead.")
@always_inline("nodebug")
fn simdbytewidth[target: _TargetType = _current_target()]() -> Int:
    return simd_byte_width[target]()


@always_inline("nodebug")
fn size_of[type: AnyType, target: _TargetType = _current_target()]() -> Int:
    """Returns the size of (in bytes) of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The size of the type in bytes.

    Example:
    ```mojo
    from sys.info import size_of
    def main():
        print(
            size_of[UInt8]() == 1,
            size_of[UInt16]() == 2,
            size_of[Int32]() == 4,
            size_of[Float64]() == 8,
            size_of[
                SIMD[DType.uint8, 4]
            ]() == 4,
        )
    ```
    Note: `align_of` is in same module.
    """
    alias mlir_type = __mlir_attr[
        `#kgen.param.expr<rebind, #kgen.type<!kgen.param<`,
        type,
        `>> : `,
        AnyType,
        `> : !kgen.type`,
    ]
    return Int(
        __mlir_attr[
            `#kgen.param.expr<get_sizeof, #kgen.type<`,
            mlir_type,
            `> : !kgen.type,`,
            target,
            `> : index`,
        ]
    )


@deprecated("Use `sys.size_of()` instead.")
@always_inline("nodebug")
fn sizeof[type: AnyType, target: _TargetType = _current_target()]() -> Int:
    return size_of[type, target]()


@always_inline("nodebug")
fn size_of[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    """Returns the size of (in bytes) of the dtype.

    Parameters:
        dtype: The DType in question.
        target: The target architecture.

    Returns:
        The size of the dtype in bytes.
    """
    return Int(
        __mlir_attr[
            `#kgen.param.expr<get_sizeof, #kgen.type<`,
            Scalar[dtype]._mlir_type,
            `> : !kgen.type,`,
            target,
            `> : index`,
        ]
    )


@deprecated("Use `sys.size_of()` instead.")
@always_inline("nodebug")
fn sizeof[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    return size_of[dtype, target]()


@always_inline("nodebug")
fn align_of[type: AnyType, target: _TargetType = _current_target()]() -> Int:
    """Returns the align of (in bytes) of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The alignment of the type in bytes.
    """
    alias mlir_type = __mlir_attr[
        `#kgen.param.expr<rebind, #kgen.type<!kgen.param<`,
        type,
        `>> : `,
        AnyType,
        `> : !kgen.type`,
    ]
    return Int(
        __mlir_attr[
            `#kgen.param.expr<get_alignof, #kgen.type<`,
            +mlir_type,
            `> : !kgen.type,`,
            target,
            `> : index`,
        ]
    )


@deprecated("Use `sys.align_of()` instead.")
@always_inline("nodebug")
fn alignof[type: AnyType, target: _TargetType = _current_target()]() -> Int:
    return align_of[type, target]()


@always_inline("nodebug")
fn align_of[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    """Returns the align of (in bytes) of the dtype.

    Parameters:
        dtype: The DType in question.
        target: The target architecture.

    Returns:
        The alignment of the dtype in bytes.
    """
    return Int(
        __mlir_attr[
            `#kgen.param.expr<get_alignof, #kgen.type<`,
            Scalar[dtype]._mlir_type,
            `> : !kgen.type,`,
            target,
            `> : index`,
        ]
    )


@deprecated("Use `sys.align_of()` instead.")
@always_inline("nodebug")
fn alignof[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    return align_of[dtype, target]()


@always_inline("nodebug")
fn bit_width_of[
    type: AnyTrivialRegType, target: _TargetType = _current_target()
]() -> Int:
    """Returns the size of (in bits) of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The size of the type in bits.
    """
    alias CHAR_BIT = 8
    return CHAR_BIT * size_of[type, target=target]()


@deprecated("Use `sys.bit_width_of()` instead.")
@always_inline("nodebug")
fn bitwidthof[
    type: AnyTrivialRegType, target: _TargetType = _current_target()
]() -> Int:
    return bit_width_of[type, target]()


@always_inline("nodebug")
fn bit_width_of[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    """Returns the size of (in bits) of the dtype.

    Parameters:
        dtype: The type in question.
        target: The target architecture.

    Returns:
        The size of the dtype in bits.
    """
    return bit_width_of[Scalar[dtype]._mlir_type, target=target]()


@deprecated("Use `sys.bit_width_of()` instead.")
@always_inline("nodebug")
fn bitwidthof[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    return bit_width_of[dtype, target]()


@always_inline("nodebug")
fn simd_width_of[
    type: AnyTrivialRegType, target: _TargetType = _current_target()
]() -> Int:
    """Returns the vector size of the type on the host system.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The vector size of the type on the host system.
    """
    return simd_bit_width[target]() // bit_width_of[type, target]()


@deprecated("Use `sys.simd_width_of()` instead.")
@always_inline("nodebug")
fn simdwidthof[
    type: AnyTrivialRegType, target: _TargetType = _current_target()
]() -> Int:
    return simd_width_of[type, target]()


@always_inline("nodebug")
fn simd_width_of[
    dtype: DType, target: _TargetType = _current_target()
]() -> Int:
    """Returns the vector size of the type on the host system.

    Parameters:
        dtype: The DType in question.
        target: The target architecture.

    Returns:
        The vector size of the dtype on the host system.
    """
    return simd_width_of[Scalar[dtype]._mlir_type, target]()


@always_inline("nodebug")
fn num_physical_cores() -> Int:
    """Returns the number of physical cores across all CPU sockets.


    Returns:
        Int: The number of physical cores on the system.
    """
    return _external_call_const["KGEN_CompilerRT_NumPhysicalCores", Int]()


@always_inline
fn num_logical_cores() -> Int:
    """Returns the number of hardware threads, including hyperthreads across all
    CPU sockets.

    Returns:
        Int: The number of threads on the system.
    """
    return _external_call_const["KGEN_CompilerRT_NumLogicalCores", Int]()


@always_inline
fn num_performance_cores() -> Int:
    """Returns the number of physical performance cores across all CPU sockets.
    If not known, returns the total number of physical cores.

    Returns:
        Int: The number of physical performance cores on the system.
    """
    return _external_call_const["KGEN_CompilerRT_NumPerformanceCores", Int]()


@always_inline
fn _macos_version() raises -> Tuple[Int, Int, Int]:
    """Gets the macOS version.

    Returns:
        The version triple of macOS.
    """

    constrained[
        CompilationTarget.is_macos(),
        "the operating system must be macOS",
    ]()

    alias INITIAL_CAPACITY = 32

    # Overallocate the string.
    var buf_len = Int(INITIAL_CAPACITY)
    var osver = String(unsafe_uninit_length=UInt(buf_len))

    var err = external_call["sysctlbyname", Int32](
        "kern.osproductversion".unsafe_cstr_ptr(),
        osver.unsafe_ptr(),
        Pointer(to=buf_len),
        OpaquePointer(),
        Int(0),
    )
    if err:
        raise "Unable to query macOS version"

    # Truncate the string down to the actual length.
    osver = osver[0:buf_len]

    var major = 0
    var minor = 0
    var patch = 0

    if "." in osver:
        major = Int(osver[: osver.find(".")])
        osver = osver[osver.find(".") + 1 :]

    if "." in osver:
        minor = Int(osver[: osver.find(".")])
        osver = osver[osver.find(".") + 1 :]

    if "." in osver:
        patch = Int(osver[: osver.find(".")])

    return (major, minor, patch)


# ===-----------------------------------------------------------------------===#
# Detect GPU on host side
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn has_accelerator() -> Bool:
    """Returns True if the host system has an accelerator and False otherwise.

    Returns:
        True if the host system has an accelerator.
    """
    return is_gpu() or _accelerator_arch() != ""


@always_inline("nodebug")
fn has_amd_gpu_accelerator() -> Bool:
    """Returns True if the host system has an AMD GPU and False otherwise.

    Returns:
        True if the host system has an AMD GPU.
    """
    return is_amd_gpu() or "amd" in _accelerator_arch()


@always_inline("nodebug")
fn has_nvidia_gpu_accelerator() -> Bool:
    """Returns True if the host system has an NVIDIA GPU and False otherwise.

    Returns:
        True if the host system has an NVIDIA GPU.
    """
    return is_nvidia_gpu() or "nvidia" in _accelerator_arch()


@always_inline("nodebug")
fn has_apple_gpu_accelerator() -> Bool:
    """Returns True if the host system has a Metal GPU and False otherwise.
    Returns:
        True if the host system has a Metal GPU.
    """
    return is_apple_gpu() or "metal" in _accelerator_arch()
