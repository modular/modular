# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for working with the
warp-matrix-matrix-multiplication (wmma) instructions."""

from sys import _RegisterPackType, llvm_intrinsic, sizeof
from sys._assembly import inlined_assembly

from gpu.memory import AddressSpace
from memory import UnsafePointer, bitcast

from utils import StaticTuple


@always_inline
fn mma(inout d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs warp sync Tensor Core based Matrix-multiply and accumulate(MMA) operation.
    """

    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # ===------------------------------------------------------------------===#
    @parameter
    if (
        d.type is DType.float16
        and d.size == 4
        and a.type is DType.float16
        and a.size == 4
        and b.type is DType.float16
        and b.size == 2
        and c.type is DType.float16
        and c.size == 4
    ):
        var sa = a.split()
        var sc = c.split()

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f16.f16",
            _RegisterPackType[SIMD[DType.float16, 2], SIMD[DType.float16, 2]],
        ](sa[0], sa[1], b, sc[0], sc[1])

        d = rebind[__type_of(d)](r[0].join(r[1]))
    elif (
        d.type is DType.float16
        and d.size == 2
        and a.type is DType.float16
        and a.size == 1
        and b.type is DType.float16
        and b.size == 1
        and c.type is DType.float16
        and c.size == 2
    ):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m8n8k4.row.col.f16.f16",
            _RegisterPackType[Float16, Float16],
        ](a, b, c)
        d = rebind[__type_of(d)](r[0].join(r[1]))

    # ===------------------------------------------------------------------===#
    # F32 = F16 * F16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float16
        and a.size == 4
        and b.type is DType.float16
        and b.size == 2
        and c.type is DType.float32
        and c.size == 4
    ):
        var sa = a.split()
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f32.f32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            sa[0],
            sa[1],
            b,
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )

        d = rebind[__type_of(d)](SIMD[DType.float32, 4](r[0], r[1], r[2], r[3]))
    elif (
        d.type is DType.float32
        and d.size == 2
        and a.type is DType.float16
        and a.size == 1
        and b.type is DType.float16
        and b.size == 1
        and c.type is DType.float32
        and c.size == 2
    ):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m8n8k4.row.col.f32.f32",
            _RegisterPackType[Float32, Float32],
        ](a, b, c)
        d = rebind[__type_of(d)](r[0].join(r[1]))

    # ===------------------------------------------------------------------===#
    # F32 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.bfloat16
        and a.size == 4
        and b.type is DType.bfloat16
        and b.size == 2
        and c.type is DType.float32
        and c.size == 4
    ):
        var sa = a.split()
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.bf16",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            bitcast[DType.int32, 1](sa[0]),
            bitcast[DType.int32, 1](sa[1]),
            bitcast[DType.int32, 1](b),
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[__type_of(d)](SIMD[DType.float32, 4](r[0], r[1], r[2], r[3]))

    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.bfloat16
        and a.size == 8
        and b.type is DType.bfloat16
        and b.size == 4
        and c.type is DType.float32
        and c.size == 4
    ):
        var sa = a.split()
        var sa1 = sa[0].split()
        var sa2 = sa[1].split()
        var sb = b.split()
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k16.row.col.bf16",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            bitcast[DType.int32, 1](sa1[0]),
            bitcast[DType.int32, 1](sa1[1]),
            bitcast[DType.int32, 1](sa2[0]),
            bitcast[DType.int32, 1](sa2[1]),
            bitcast[DType.int32, 1](sb[0]),
            bitcast[DType.int32, 1](sb[1]),
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[__type_of(d)](SIMD[DType.float32, 4](r[0], r[1], r[2], r[3]))

    # ===------------------------------------------------------------------===#
    # F32 = tf32 * tf32 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float32
        and a.size == 2
        and b.type is DType.float32
        and b.size == 1
        and c.type is DType.float32
        and c.size == 4
    ):
        var a0 = bitcast[DType.uint32, 2](a)
        var b0 = bitcast[DType.uint32, 1](b)
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k4.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a0[0],
            a0[1],
            b0,
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[__type_of(d)](SIMD[DType.float32, 4](r[0], r[1], r[2], r[3]))

    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float32
        and a.size == 4
        and b.type is DType.float32
        and b.size == 2
        and c.type is DType.float32
        and c.size == 4
    ):
        var a0 = bitcast[DType.uint32, 4](a)
        var b0 = bitcast[DType.uint32, 2](b)
        var c0 = bitcast[DType.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a0[0],
            a0[1],
            a0[2],
            a0[3],
            b0[0],
            b0[1],
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[__type_of(d)](SIMD[DType.float32, 4](r[0], r[1], r[2], r[3]))

    # ===------------------------------------------------------------------===#
    # F32 = FP8 * FP8 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float8e4m3
        and a.size == 16
        and b.type is DType.float8e4m3
        and b.size == 8
        and c.type is DType.float32
        and c.size == 4
    ):
        var a0 = bitcast[DType.uint32, 4](a)
        var b0 = bitcast[DType.uint32, 2](b)

        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0, $1,"
                " $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r",
        ](
            a0[0],
            a0[1],
            a0[2],
            a0[3],
            b0[0],
            b0[1],
            c[0],
            c[1],
            c[2],
            c[3],
        )
        d = rebind[__type_of(d)](SIMD[DType.float32, 4](r[0], r[1], r[2], r[3]))
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float8e5m2
        and a.size == 16
        and b.type is DType.float8e5m2
        and b.size == 8
        and c.type is DType.float32
        and c.size == 4
    ):
        var a0 = bitcast[DType.uint32, 4](a)
        var b0 = bitcast[DType.uint32, 2](b)

        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {$0, $1,"
                " $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r",
        ](
            a0[0],
            a0[1],
            a0[2],
            a0[3],
            b0[0],
            b0[1],
            c[0],
            c[1],
            c[2],
            c[3],
        )
        d = rebind[__type_of(d)](SIMD[DType.float32, 4](r[0], r[1], r[2], r[3]))

    else:
        constrained[False, "no valid implementation of mma"]()


# ===------------------------------------------------------------------===#
# LDMATRIX Instruction
# ===------------------------------------------------------------------===#


@always_inline
fn ld_matrix[
    type: DType, simd_width: Int, transpose: Bool = False
](ptr: UnsafePointer[Scalar[type], AddressSpace.SHARED]) -> SIMD[
    type, simd_width
]:
    """Performs warp sync copy from shared memory to registers.
    Loads in a fashion that can be used directly by tensor core MMA instructions.
    """

    # TODO: Investigate if fp8 can work with transposed ld_matrix.
    constrained[
        (transpose and type.is_half_float()) or (not transpose),
        "Transposed ld_matrix is only for half precision.",
    ]()

    # The register width is fixed at 4 Bytes (32 bits)
    alias register_btypes = 4
    alias register_width = register_btypes // sizeof[type]()
    alias num_registers = simd_width // register_width

    # Full intrinsic is base + suffix
    alias base = "llvm.nvvm.ldmatrix.sync.aligned.m8n8"

    @parameter
    fn get_suffix() -> StringLiteral:
        alias sfx = ".b16.p3"
        if transpose:
            return ".trans" + sfx
        return sfx

    var d = SIMD[type, simd_width]()

    # Here .x1 means every thread would use a single register, x2 is 2 while x4 is 4 registers
    # An mma of shape m16n8k8 of type TF32 means for Matrix A every thread would have 4 registers hence .x4
    # and input simd_width being equal to 4
    @parameter
    if num_registers == 1:
        alias ins = base + ".x1" + get_suffix()
        var r = llvm_intrinsic[ins, UInt32](ptr)
        var r0 = bitcast[type, register_width](r[0])

        d = rebind[SIMD[type, simd_width]](r0)

    elif num_registers == 2:
        alias ins = base + ".x2" + get_suffix()
        var r = llvm_intrinsic[ins, _RegisterPackType[UInt32, UInt32]](ptr)
        var r0 = bitcast[type, register_width](r[0])
        var r1 = bitcast[type, register_width](r[1])

        d = rebind[SIMD[type, simd_width]](r0.join(r1))

    else:
        constrained[
            num_registers == 4,
            "no valid implementation of ldmatrix instruction",
        ]()
        alias ins = base + ".x4" + get_suffix()
        var r = llvm_intrinsic[
            ins, _RegisterPackType[UInt32, UInt32, UInt32, UInt32]
        ](ptr)

        # Unpack result to 4 vectors (one per register), then concat them to return.
        var r0 = bitcast[type, register_width](r[0])
        var r1 = bitcast[type, register_width](r[1])
        var r2 = bitcast[type, register_width](r[2])
        var r3 = bitcast[type, register_width](r[3])
        d = rebind[SIMD[type, simd_width]](r0.join(r1).join(r2.join(r3)))

        # The following creates additional copies uint32 <-> 2xbf16 in matmul.
        # @parameter
        # for i in range(num_registers):
        #     var vec_per_register = bitcast[type, register_width](
        #         rebind[UInt32](r[i])
        #     )

        #     @parameter
        #     for j in range(register_width):
        #         d[i * register_width + j] = vec_per_register[j]

    return d


# ===------------------------------------------------------------------===#
# Warp group MMA asynchronous compute and synchronization instructions.
# ===------------------------------------------------------------------===#


# Shared memory operand descriptor.
@register_passable("trivial")
struct WGMMADescriptor[dtype: DType]:
    # The bits as the following.
    # start address : 14-bit
    # leading byte_offset: 14 bit
    # stride byte offset: 14 bit
    # base offset: 3 bit
    # swizzle mode: 2 bit
    #  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
    #  |   0-13  |14-15|   16-29   |30-31|   32-45   |46-48|49-51|   52-61   |62-63|
    #  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
    #  |  14bits |2bits|   14bits  |2bits|   14bits  |2bits|3bits|   10bits  |2bits|
    #  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
    #  | BaseAddr|  0  | LeadingDim|  0  |   Stride  |  0  |Offst|     0     |Swzle|
    #  +---------+-----+-----------+-----+-----------+-----+-----+-----------+-----+
    # See:
    # # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor

    var desc: Int64

    @implicit
    fn __init__(out self, val: Int64):
        self.desc = val

    @staticmethod
    fn create[
        stride_byte_offset: Int, leading_byte_offset: Int, swizzle_mode: Int = 0
    ](
        smem_ptr: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.SHARED
        ],
    ) -> Self:
        @parameter
        fn insert_bit[start_bit: Int](target: Int64, val: Int64) -> Int64:
            return target | (val << start_bit)

        var swizzle = Int64(swizzle_mode)
        var offset = Int64(0)
        var stride_dim = Int64(stride_byte_offset)
        var lead_dim = Int64(leading_byte_offset)

        var base_ptr = int(smem_ptr)
        var start_address = (base_ptr >> 4)

        var desc = Int64(0)
        # bits [63, 62] swizzle type
        desc = insert_bit[62](desc, swizzle)
        # bits [51 .. 49] offset
        desc = insert_bit[49](desc, offset)
        # bits [48 .. 32]
        desc = insert_bit[32](desc, stride_dim)  # bits [32-45]
        desc = insert_bit[16](desc, lead_dim)  # bits [16-29]
        desc = insert_bit[0](desc, start_address)  # bits [0-13]

        return desc


@always_inline
fn wgmma_fence_aligned():
    __mlir_op.`nvvm.wgmma.fence.aligned`[_type=None]()


@always_inline
fn wgmma_commit_group_sync():
    __mlir_op.`nvvm.wgmma.commit.group.sync.aligned`[_type=None]()


@always_inline
fn wgmma_wait_group_sync():
    __mlir_op.`nvvm.wgmma.wait.group.sync.aligned`[
        _properties = __mlir_attr.`{group = 0 : i32}`, _type=None
    ]()


@always_inline
fn wgmma_async[
    m: Int,
    n: Int,
    k: Int,
    c_dtype: DType,
    width: Int,
    /,
    *,
    a_type: DType,
    b_type: DType,
    layout_a: StringLiteral = "row",
    layout_b: StringLiteral = "col",
](
    mat_a_desc: WGMMADescriptor,
    mat_b_desc: WGMMADescriptor,
    c_reg: SIMD[c_dtype, width],
) -> __type_of(c_reg):
    """Performs warp group async Matrix-multiply and accumulate(WGMMA) operation.
    """
    var desc_a_value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i64](
        mat_a_desc.desc.value
    )
    var desc_b_value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i64](
        mat_b_desc.desc.value
    )

    @parameter
    if a_type is DType.tensor_float32:
        return __mlir_op.`pop.nvvm.wgmma.mma_async`[
            shape_m = m.value,
            shape_n = n.value,
            shape_k = k.value,
            type_a = __mlir_attr.`tf32`,
            type_b = __mlir_attr.`tf32`,
            type_c = __mlir_attr.`f32`,
            layout_a = layout_a.value,
            layout_b = layout_b.value,
            _type = __type_of(c_reg.value),
        ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.bfloat16:
        return __mlir_op.`pop.nvvm.wgmma.mma_async`[
            shape_m = m.value,
            shape_n = n.value,
            shape_k = k.value,
            type_a = __mlir_attr.`bf16`,
            type_b = __mlir_attr.`bf16`,
            type_c = __mlir_attr.`f32`,
            layout_a = layout_a.value,
            layout_b = layout_b.value,
            _type = __type_of(c_reg.value),
        ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.float16:
        if c_dtype is DType.uint32:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f16`,
                type_b = __mlir_attr.`f16`,
                type_c = __mlir_attr.`f16`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
        else:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f16`,
                type_b = __mlir_attr.`f16`,
                type_c = __mlir_attr.`f32`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.float8e4m3 and b_type is DType.float8e4m3:
        if c_dtype is DType.uint32:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E4M3`,
                type_b = __mlir_attr.`f8E4M3`,
                type_c = __mlir_attr.`f16`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
        else:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E4M3`,
                type_b = __mlir_attr.`f8E4M3`,
                type_c = __mlir_attr.`f32`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.float8e5m2 and b_type is DType.float8e5m2:
        if c_dtype is DType.uint32:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E5M2`,
                type_b = __mlir_attr.`f8E5M2`,
                type_c = __mlir_attr.`f16`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
        else:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E5M2`,
                type_b = __mlir_attr.`f8E5M2`,
                type_c = __mlir_attr.`f32`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.float8e4m3 and b_type is DType.float8e5m2:
        if c_dtype is DType.uint32:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E4M3`,
                type_b = __mlir_attr.`f8E5M2`,
                type_c = __mlir_attr.`f16`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
        else:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E4M3`,
                type_b = __mlir_attr.`f8E5M2`,
                type_c = __mlir_attr.`f32`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.float8e5m2 and b_type is DType.float8e4m3:
        if c_dtype is DType.uint32:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E5M2`,
                type_b = __mlir_attr.`f8E4M3`,
                type_c = __mlir_attr.`f16`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
        else:
            return __mlir_op.`pop.nvvm.wgmma.mma_async`[
                shape_m = m.value,
                shape_n = n.value,
                shape_k = k.value,
                type_a = __mlir_attr.`f8E5M2`,
                type_b = __mlir_attr.`f8E4M3`,
                type_c = __mlir_attr.`f32`,
                layout_a = layout_a.value,
                layout_b = layout_b.value,
                _type = __type_of(c_reg.value),
            ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.int8 and b_type is DType.int8:
        return __mlir_op.`pop.nvvm.wgmma.mma_async`[
            shape_m = m.value,
            shape_n = n.value,
            shape_k = k.value,
            type_a = __mlir_attr.`si8`,
            type_b = __mlir_attr.`si8`,
            type_c = __mlir_attr.`si32`,
            layout_a = layout_a.value,
            layout_b = layout_b.value,
            _type = __type_of(c_reg.value),
        ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.uint8 and b_type is DType.uint8:
        return __mlir_op.`pop.nvvm.wgmma.mma_async`[
            shape_m = m.value,
            shape_n = n.value,
            shape_k = k.value,
            type_a = __mlir_attr.`ui8`,
            type_b = __mlir_attr.`ui8`,
            type_c = __mlir_attr.`si32`,
            layout_a = layout_a.value,
            layout_b = layout_b.value,
            _type = __type_of(c_reg.value),
        ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.int8 and b_type is DType.uint8:
        return __mlir_op.`pop.nvvm.wgmma.mma_async`[
            shape_m = m.value,
            shape_n = n.value,
            shape_k = k.value,
            type_a = __mlir_attr.`si8`,
            type_b = __mlir_attr.`ui8`,
            type_c = __mlir_attr.`si32`,
            layout_a = layout_a.value,
            layout_b = layout_b.value,
            _type = __type_of(c_reg.value),
        ](desc_a_value, desc_b_value, c_reg.value)
    elif a_type is DType.uint8 and b_type is DType.int8:
        return __mlir_op.`pop.nvvm.wgmma.mma_async`[
            shape_m = m.value,
            shape_n = n.value,
            shape_k = k.value,
            type_a = __mlir_attr.`ui8`,
            type_b = __mlir_attr.`si8`,
            type_c = __mlir_attr.`si32`,
            layout_a = layout_a.value,
            layout_b = layout_b.value,
            _type = __type_of(c_reg.value),
        ](desc_a_value, desc_b_value, c_reg.value)
    return c_reg
