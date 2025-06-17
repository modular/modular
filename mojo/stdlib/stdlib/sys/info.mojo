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

    This struct encapsulates various methods to query target-specific information
    such as architecture features, OS details, endianness, and memory characteristics.

    Parameters:
        value: The target architecture to query. Defaults to the current target.
    """

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

    @staticmethod
    fn has_sse4() -> Bool:
        """Checks if the target supports SSE4 instructions.

        Returns:
            True if the target supports SSE4, False otherwise.
        """
        return Self._has_feature["sse4"]()

    # Platforms

    @staticmethod
    fn is_x86() -> Bool:
        """Checks if the target is an x86 architecture.

        Returns:
            True if the target is x86, False otherwise.
        """
        return Self.has_sse4()


@always_inline("nodebug")
fn _accelerator_arch() -> StaticString:
    return __mlir_attr.`#kgen.param.expr<accelerator_arch> : !kgen.string`


fn _get_arch[target: __mlir_type.`!kgen.target`]() -> StaticString:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        target,
        `, "arch" : !kgen.string`,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
fn _current_arch_kgen() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        _current_target(),
        `, "arch" : !kgen.string`,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
fn _current_arch() -> StaticString:
    return _current_arch_kgen()


@always_inline("nodebug")
@deprecated("Use `CompilationTarget.is_x86()` instead.")
fn is_x86() -> Bool:
    """Returns True if the host system architecture is X86 and False otherwise.

    Returns:
        True if the host system architecture is X86 and False otherwise.
    """
    return CompilationTarget.has_sse4()


@always_inline("nodebug")
@deprecated("Use `CompilationTarget.has_sse4()` instead.")
fn has_sse4() -> Bool:
    """Returns True if the host system has sse4, otherwise returns False.

    Returns:
        True if the host system has sse4, otherwise returns False.
    """
    return CompilationTarget.has_sse4()


@always_inline("nodebug")
fn has_avx() -> Bool:
    """Returns True if the host system has AVX, otherwise returns False.

    Returns:
        True if the host system has AVX, otherwise returns False.
    """
    return __mlir_attr[
        `#kgen.param.expr<target_has_feature,`,
        _current_target(),
        `, "avx" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn has_avx2() -> Bool:
    """Returns True if the host system has AVX2, otherwise returns False.

    Returns:
        True if the host system has AVX2, otherwise returns False.
    """
    return __mlir_attr[
        `#kgen.param.expr<target_has_feature,`,
        _current_target(),
        `, "avx2" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn has_avx512f() -> Bool:
    """Returns True if the host system has AVX512, otherwise returns False.

    Returns:
        True if the host system has AVX512, otherwise returns False.
    """
    return __mlir_attr[
        `#kgen.param.expr<target_has_feature,`,
        _current_target(),
        `, "avx512f" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn has_fma() -> Bool:
    """Returns True if the host system has FMA (Fused Multiply-Add) support,
    otherwise returns False.

    Returns:
        True if the host system has FMA support, otherwise returns False.
    """
    return __mlir_attr[
        `#kgen.param.expr<target_has_feature,`,
        _current_target(),
        `, "fma" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn has_vnni() -> Bool:
    """Returns True if the host system has avx512_vnni, otherwise returns False.

    Returns:
        True if the host system has avx512_vnni, otherwise returns False.
    """
    return (
        __mlir_attr[
            `#kgen.param.expr<target_has_feature,`,
            _current_target(),
            `, "avx512vnni" : !kgen.string`,
            `> : i1`,
        ]
        or __mlir_attr[
            `#kgen.param.expr<target_has_feature,`,
            _current_target(),
            `, "avxvnni" : !kgen.string`,
            `> : i1`,
        ]
    )


@always_inline("nodebug")
fn has_neon() -> Bool:
    """Returns True if the host system has Neon support, otherwise returns
    False.

    Returns:
        True if the host system support the Neon instruction set.
    """
    alias neon_flag: Bool = __mlir_attr[
        `#kgen.param.expr<target_has_feature,`,
        _current_target(),
        `, "neon" : !kgen.string`,
        `> : i1`,
    ]

    @parameter
    if neon_flag:
        return True
    return is_apple_silicon()


@always_inline("nodebug")
fn has_neon_int8_dotprod() -> Bool:
    """Returns True if the host system has the Neon int8 dot product extension,
    otherwise returns False.

    Returns:
        True if the host system support the Neon int8 dot product extension and
        False otherwise.
    """
    return (
        has_neon()
        and __mlir_attr[
            `#kgen.param.expr<target_has_feature,`,
            _current_target(),
            `, "dotprod" : !kgen.string`,
            `> : i1`,
        ]
    )


@always_inline("nodebug")
fn has_neon_int8_matmul() -> Bool:
    """Returns True if the host system has the Neon int8 matrix multiplication
    extension (I8MM), otherwise returns False.

    Returns:
        True if the host system support the Neon int8 matrix multiplication
        extension (I8MM) and False otherwise.
    """
    return (
        has_neon()
        and __mlir_attr[
            `#kgen.param.expr<target_has_feature,`,
            _current_target(),
            `, "i8mm" : !kgen.string`,
            `> : i1`,
        ]
    )


@always_inline("nodebug")
fn is_apple_m1() -> Bool:
    """Returns True if the host system is an Apple M1 with AMX support,
    otherwise returns False.

    Returns:
        True if the host system is an Apple M1 with AMX support and False
        otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _current_arch_kgen(),
        `, "apple-m1" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn is_apple_m2() -> Bool:
    """Returns True if the host system is an Apple M2 with AMX support,
    otherwise returns False.

    Returns:
        True if the host system is an Apple M2 with AMX support and False
        otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _current_arch_kgen(),
        `, "apple-m2" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn is_apple_m3() -> Bool:
    """Returns True if the host system is an Apple M3 with AMX support,
    otherwise returns False.

    Returns:
        True if the host system is an Apple M3 with AMX support and False
        otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _current_arch_kgen(),
        `, "apple-m3" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn is_apple_m4() -> Bool:
    """Returns True if the host system is an Apple M4 with AMX support,
    otherwise returns False.

    Returns:
        True if the host system is an Apple M4 with AMX support and False
        otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _current_arch_kgen(),
        `, "apple-m4" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn is_apple_silicon() -> Bool:
    """Returns True if the host system is an Apple Silicon with AMX support,
    otherwise returns False.

    Returns:
        True if the host system is an Apple Silicon with AMX support and False
        otherwise.
    """
    return is_apple_m1() or is_apple_m2() or is_apple_m3() or is_apple_m4()


@always_inline("nodebug")
fn is_neoverse_n1() -> Bool:
    """Returns True if the host system is a Neoverse N1 system, otherwise
    returns False.

    Returns:
        True if the host system is a Neoverse N1 system and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _current_arch_kgen(),
        `, "neoverse-n1" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn has_intel_amx() -> Bool:
    """Returns True if the host system has Intel AMX support, otherwise returns
    False.

    Returns:
        True if the host system has Intel AMX and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<target_has_feature,`,
        _current_target(),
        `, "amx-tile" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn _os_attr() -> StaticString:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        _current_target(),
        `, "os" : !kgen.string`,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
fn os_is_macos() -> Bool:
    """Returns True if the host operating system is macOS.

    Returns:
        True if the host operating system is macOS and False otherwise.
    """
    return (
        __mlir_attr[
            `#kgen.param.expr<eq,`,
            _get_kgen_string[_os_attr()](),
            `,`,
            `"darwin" : !kgen.string`,
            `> : i1`,
        ]
        or __mlir_attr[
            `#kgen.param.expr<eq,`,
            _get_kgen_string[_os_attr()](),
            `,`,
            `"macosx" : !kgen.string`,
            `> : i1`,
        ]
    )


@always_inline("nodebug")
fn os_is_linux() -> Bool:
    """Returns True if the host operating system is Linux.

    Returns:
        True if the host operating system is Linux and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _get_kgen_string[_os_attr()](),
        `,`,
        `"linux" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn os_is_windows() -> Bool:
    """Returns True if the host operating system is Windows.

    Returns:
        True if the host operating system is Windows and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        _get_kgen_string[_os_attr()](),
        `,`,
        `"windows" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn _triple_attr[
    triple: __mlir_type.`!kgen.target` = _current_target()
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        triple,
        `, "triple" : !kgen.string`,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
fn is_triple[
    name: StringLiteral, target: __mlir_type.`!kgen.target` = _current_target()
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
    return is_nvidia_gpu() and _current_arch() == subarch


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
    """Returns True if the target triple is GPU and  False otherwise.

    Returns:
        True if the triple target is GPU and False otherwise.
    """
    return is_nvidia_gpu() or is_amd_gpu()


@always_inline("nodebug")
fn is_little_endian[
    target: __mlir_type.`!kgen.target` = _current_target()
]() -> Bool:
    """Returns True if the host endianness is little and False otherwise.

    Parameters:
        target: The target architecture.

    Returns:
        True if the host target is little endian and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            _current_target(),
            `, "endianness" : !kgen.string`,
            `> : !kgen.string`,
        ],
        `,`,
        `"little" : !kgen.string`,
        `> : i1`,
    ]


@always_inline("nodebug")
fn is_big_endian[
    target: __mlir_type.`!kgen.target` = _current_target()
]() -> Bool:
    """Returns True if the host endianness is big and False otherwise.

    Parameters:
        target: The target architecture.

    Returns:
        True if the host target is big endian and False otherwise.
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
fn is_32bit[target: __mlir_type.`!kgen.target` = _current_target()]() -> Bool:
    """Returns True if the maximum integral value is 32 bit.

    Parameters:
        target: The target architecture.

    Returns:
        True if the maximum integral value is 32 bit, False otherwise.
    """
    return sizeof[DType.index, target]() == sizeof[DType.int32, target]()


@always_inline("nodebug")
fn is_64bit[target: __mlir_type.`!kgen.target` = _current_target()]() -> Bool:
    """Returns True if the maximum integral value is 64 bit.

    Parameters:
        target: The target architecture.

    Returns:
        True if the maximum integral value is 64 bit, False otherwise.
    """
    return sizeof[DType.index, target]() == sizeof[DType.int64, target]()


@always_inline("nodebug")
fn simdbitwidth[
    target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
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


@always_inline("nodebug")
fn simdbytewidth[
    target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
    """Returns the vector size (in bytes) of the specified target.

    Parameters:
        target: The target architecture.

    Returns:
        The vector size (in bytes) of the host system.
    """
    alias CHAR_BIT = 8
    return simdbitwidth[target]() // CHAR_BIT


@always_inline("nodebug")
fn sizeof[
    type: AnyType, target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
    """Returns the size of (in bytes) of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The size of the type in bytes.

    Example:
    ```mojo
    from sys.info import sizeof
    def main():
        print(
            sizeof[UInt8]() == 1,
            sizeof[UInt16]() == 2,
            sizeof[Int32]() == 4,
            sizeof[Float64]() == 8,
            sizeof[
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


@always_inline("nodebug")
fn sizeof[
    dtype: DType, target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
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
            `!pop.scalar<`,
            dtype.value,
            `>`,
            `> : !kgen.type,`,
            target,
            `> : index`,
        ]
    )


@always_inline("nodebug")
fn alignof[
    type: AnyType, target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
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


@always_inline("nodebug")
fn alignof[
    dtype: DType, target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
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
            `!pop.scalar<`,
            dtype.value,
            `>`,
            `> : !kgen.type,`,
            target,
            `> : index`,
        ]
    )


@always_inline("nodebug")
fn bitwidthof[
    type: AnyTrivialRegType,
    target: __mlir_type.`!kgen.target` = _current_target(),
]() -> Int:
    """Returns the size of (in bits) of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The size of the type in bits.
    """
    alias CHAR_BIT = 8
    return CHAR_BIT * sizeof[type, target=target]()


@always_inline("nodebug")
fn bitwidthof[
    dtype: DType, target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
    """Returns the size of (in bits) of the dtype.

    Parameters:
        dtype: The type in question.
        target: The target architecture.

    Returns:
        The size of the dtype in bits.
    """
    return bitwidthof[
        __mlir_type[`!pop.scalar<`, dtype.value, `>`], target=target
    ]()


@always_inline("nodebug")
fn simdwidthof[
    type: AnyTrivialRegType,
    target: __mlir_type.`!kgen.target` = _current_target(),
]() -> Int:
    """Returns the vector size of the type on the host system.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The vector size of the type on the host system.
    """
    return simdbitwidth[target]() // bitwidthof[type, target]()


@always_inline("nodebug")
fn simdwidthof[
    dtype: DType, target: __mlir_type.`!kgen.target` = _current_target()
]() -> Int:
    """Returns the vector size of the type on the host system.

    Parameters:
        dtype: The DType in question.
        target: The target architecture.

    Returns:
        The vector size of the dtype on the host system.
    """
    return simdwidthof[__mlir_type[`!pop.scalar<`, dtype.value, `>`], target]()


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

    constrained[os_is_macos(), "the operating system must be macOS"]()

    alias INITIAL_CAPACITY = 32

    # Overallocate the string.
    var buf_len = Int(INITIAL_CAPACITY)
    var osver = String(unsafe_uninit_length=buf_len)

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
