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


from .memory import AddressSpace as GPUAddressSpace
from utils import StaticTuple

# ===-----------------------------------------------------------------------===#
# MLIR type conversion utils
# ===-----------------------------------------------------------------------===#


@always_inline
fn to_llvm_shared_mem_ptr[
    type: AnyType
](
    ptr: UnsafePointer[type, address_space = GPUAddressSpace.SHARED, **_]
) -> __mlir_type.`!llvm.ptr<3>`:
    """Cast shared memory pointer to LLVMPointer Type.

    Args:
        ptr: Shared memory pointer.

    Returns:
        A pointer of type !llvm.ptr<3>.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr<3>`
    ](ptr)


@always_inline
fn to_llvm_ptr[
    type: AnyType
](ptr: UnsafePointer[type]) -> __mlir_type.`!llvm.ptr`:
    """Cast a pointer to LLVMPointer Type.

    Args:
        ptr: A pointer.

    Returns:
        A pointer of type !llvm.ptr.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr`
    ](ptr)


@always_inline
fn to_i32(val: Int32) -> __mlir_type.i32:
    """Cast Scalar I32 value into MLIR i32.

    Args:
        val: Scalar I32 value.

    Returns:
       Input casted to MLIR i32 value.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.`i32`](
        val._mlir_value
    )


@always_inline
fn to_i16(val: UInt16) -> __mlir_type.i16:
    """Cast a scalar UInt16 value into MLIR i16.

    Args:
        val: Scalar I16 value.

    Returns:
       The input value cast to an MLIR i16.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.`i16`](
        val._mlir_value
    )


@always_inline
fn to_i64(val: Int64) -> __mlir_type.i64:
    """Cast Scalar I64 value into MLIR i64.

    Args:
        val: Scalar I64 value.

    Returns:
       Input casted to MLIR i64 value.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.`i64`](
        val._mlir_value
    )


alias _dtype_to_llvm_type_f8[
    dtype: DType
] = __mlir_type.`i8` if dtype is DType.float8_e3m4 or dtype is DType.float8_e4m3fn or dtype is DType.float8_e4m3fnuz or dtype is DType.float8_e5m2 or dtype is DType.float8_e5m2fnuz else __mlir_type.`!kgen.none`

alias _dtype_to_llvm_type_bf16[
    dtype: DType
] = __mlir_type.`bf16` if dtype is DType.bfloat16 else _dtype_to_llvm_type_f8[
    dtype
]

alias _dtype_to_llvm_type_f16[
    dtype: DType
] = __mlir_type.`f16` if dtype is DType.float16 else _dtype_to_llvm_type_bf16[
    dtype
]

alias _dtype_to_llvm_type_f32[
    dtype: DType
] = __mlir_type.`f32` if dtype is DType.float32 else _dtype_to_llvm_type_f16[
    dtype
]

alias _dtype_to_llvm_type_f64[
    dtype: DType
] = __mlir_type.`f64` if dtype is DType.float64 else _dtype_to_llvm_type_f32[
    dtype
]

alias _dtype_to_llvm_type_i32[
    dtype: DType
] = __mlir_type.`i32` if dtype is DType.int32 or dtype is DType.uint32 else _dtype_to_llvm_type_f64[
    dtype
]

alias _dtype_to_llvm_type_i64[
    dtype: DType
] = __mlir_type.`i64` if dtype is DType.int64 or dtype is DType.uint64 else _dtype_to_llvm_type_i32[
    dtype
]

alias dtype_to_llvm_type[dtype: DType] = _dtype_to_llvm_type_i64[dtype]

alias llvm_struct_splat[
    field_type: AnyTrivialRegType, repeat: Int
] = __mlir_type[
    `!llvm.struct<(`,
    __mlir_type[
        `!kgen.variadic_splat<`,
        field_type,
        `, `,
        repeat._mlir_value,
        `>`,
    ],
    `)>`,
]

alias kgen_struct_splat[
    field_type: AnyTrivialRegType, repeat: Int
] = __mlir_type[
    `!kgen.struct<(`,
    __mlir_type[
        `!kgen.variadic_splat<`,
        field_type,
        `, `,
        repeat._mlir_value,
        `>`,
    ],
    `)>`,
]

alias llvm_struct_dtype_splat_type[dtype: DType, n: Int] = llvm_struct_splat[
    dtype_to_llvm_type[dtype], n
]

alias kgen_struct_dtype_splat_type[dtype: DType, n: Int] = kgen_struct_splat[
    Scalar[dtype]._mlir_type, n
]


@always_inline
fn simd_to_llvm_struct[
    dtype: DType, n: Int
](simd: SIMD[dtype, n]) -> llvm_struct_dtype_splat_type[dtype, n]:
    """Repack a SIMD value to a `!llvm.struct`.

    Args:
        simd: A SIMD value.

    Returns:
        A `!llvm.struct` with the same number of fields as the SIMD value.
    """
    var llvmst = __mlir_op.`llvm.mlir.undef`[
        _type = llvm_struct_dtype_splat_type[dtype, n]
    ]()

    var st = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = kgen_struct_dtype_splat_type[dtype, n]
    ](llvmst)

    @parameter
    for i in range(n):
        var e = simd[i]
        st = __mlir_op.`kgen.struct.replace`[
            _type = kgen_struct_dtype_splat_type[dtype, n],
            index = __mlir_attr[i._mlir_value, `:index`],
        ](e, st)

    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = llvm_struct_dtype_splat_type[dtype, n]
    ](st)


@always_inline
fn llvm_struct_to_simd[
    dtype: DType, n: Int
](llvmst: llvm_struct_dtype_splat_type[dtype, n]) -> SIMD[dtype, n]:
    """Repack value of a `!llvm.struct` type to SIMD.

    Args:
        llvmst: A `!llvm.struct` value.

    Returns:
        A SIMD value with the same number of elements as the `!llvm.struct`.
    """
    var simd = SIMD[dtype, n]()
    var st = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = kgen_struct_dtype_splat_type[dtype, n]
    ](llvmst)

    @parameter
    for i in range(n):
        var e = __mlir_op.`kgen.struct.extract`[
            _type = Scalar[dtype]._mlir_type,
            index = __mlir_attr[i._mlir_value, `:index`],
        ](st)

        simd[i] = Scalar[dtype](mlir_value=e)
    return simd


@always_inline
fn array_to_llvm_struct[
    dtype: DType, n: Int
](array: StaticTuple[Scalar[dtype], n]) -> llvm_struct_dtype_splat_type[
    dtype, n
]:
    """Repack a StaticTuple value to a `!llvm.struct`.

    Args:
        array: A array value.

    Returns:
        A `!llvm.struct` with the same number of fields as the array value.
    """
    var llvmst = __mlir_op.`llvm.mlir.undef`[
        _type = llvm_struct_dtype_splat_type[dtype, n]
    ]()

    var st = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = kgen_struct_dtype_splat_type[dtype, n]
    ](llvmst)

    @parameter
    for i in range(n):
        var e = array[i]
        st = __mlir_op.`kgen.struct.replace`[
            _type = kgen_struct_dtype_splat_type[dtype, n],
            index = __mlir_attr[i._mlir_value, `:index`],
        ](e, st)

    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = llvm_struct_dtype_splat_type[dtype, n]
    ](st)


@always_inline
fn llvm_struct_to_array[
    dtype: DType, n: Int
](llvmst: llvm_struct_dtype_splat_type[dtype, n]) -> StaticTuple[
    Scalar[dtype], n
]:
    """Repack value of a `!llvm.struct` type to StaticTuple.

    Args:
        llvmst: A `!llvm.struct` value.

    Returns:
        A array value with the same number of elements as the `!llvm.struct`.
    """
    var array = StaticTuple[Scalar[dtype], n]()
    var st = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = kgen_struct_dtype_splat_type[dtype, n]
    ](llvmst)

    @parameter
    for i in range(n):
        var e = __mlir_op.`kgen.struct.extract`[
            _type = Scalar[dtype]._mlir_type,
            index = __mlir_attr[i._mlir_value, `:index`],
        ](st)

        array[i] = Scalar[dtype](mlir_value=e)
    return array
