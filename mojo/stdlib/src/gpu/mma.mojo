# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for working with the
warp-matrix-matrix-multiplication (wmma) instructions."""

from sys import _RegisterPackType, llvm_intrinsic, sizeof
from sys._assembly import inlined_assembly

from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.memory import AddressSpace
from memory import UnsafePointer, bitcast

from utils import StaticTuple
from utils.index import Index


@always_inline
fn _unsupported_mma_op(d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    constrained[
        False,
        # fmt: off
        String(
            "no valid implementation of mma for for a=",
            a.size, "x", a.type,
            ", b=", b.size, "x", b.type,
            ", c=", c.size, "x", c.type,
            ", and d=", d.size, "x", d.type,
        )
        # fmt: on
    ]()


@always_inline
fn _mma_amd(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # ===------------------------------------------------------------------===#
    @parameter
    if (
        d.type is DType.float16
        and a.type is DType.float16
        and b.type is DType.float16
        and c.type is DType.float16
    ):
        constrained[
            False, "Function mma F16 * F16 + F16 is unsupported by AMD GPUs."
        ]()
    # ===------------------------------------------------------------------===#
    # F32 = F16 * F16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float16
        and a.size == 4
        and b.type is DType.float16
        and b.size == 4
        and c.type is DType.float32
        and c.size == 4
    ):
        alias zero: UInt32 = 0
        var r = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x16f16", SIMD[c.type, c.size]
        ](a, b, c, zero, zero, zero)
        d = rebind[__type_of(d)](r)

    # ===------------------------------------------------------------------===#
    # F32 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.bfloat16
        and a.size == 4
        and b.type is DType.bfloat16
        and b.size == 4
        and c.type is DType.float32
        and c.size == 4
    ):
        alias zero: UInt32 = 0
        var r = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", SIMD[c.type, c.size]
        ](
            bitcast[DType.int16, 4](a),
            bitcast[DType.int16, 4](b),
            c,
            zero,
            zero,
            zero,
        )
        d = rebind[__type_of(d)](r)
    # ===------------------------------------------------------------------===#
    # F32 = FP32 * FP32 + FP32
    # ===------------------------------------------------------------------===#
    elif (
        d.type is DType.float32
        and d.size == 4
        and a.type is DType.float32
        and a.size == 1
        and b.type is DType.float32
        and b.size == 1
        and c.type is DType.float32
        and c.size == 4
    ):
        alias zero: UInt32 = 0
        var r = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x4f32", SIMD[c.type, c.size]
        ](a, b, c, zero, zero, zero)
        d = rebind[__type_of(d)](r)
    else:
        _unsupported_mma_op(d, a, b, c)


@always_inline
fn _mma_nvidia(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
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
        and a.type is DType.float8_e4m3fn
        and a.size == 16
        and b.type is DType.float8_e4m3fn
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
        and a.type is DType.float8_e5m2
        and a.size == 16
        and b.type is DType.float8_e5m2
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
        _unsupported_mma_op(d, a, b, c)


@always_inline
fn mma(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs warp sync Tensor Core based Matrix-multiply and accumulate(MMA) operation.
    """

    @parameter
    if is_nvidia_gpu():
        _mma_nvidia(d, a, b, c)
    else:
        _mma_amd(d, a, b, c)


# ===------------------------------------------------------------------===#
# LDMATRIX Instruction
# ===------------------------------------------------------------------===#


@always_inline
fn ld_matrix[
    type: DType, //, simd_width: Int, *, transpose: Bool = False
](
    ptr: UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED]
) -> SIMD[type, simd_width]:
    """Performs warp sync copy from shared memory to registers.
    Loads in a fashion that can be used directly by tensor core MMA instructions.
    """

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

    @always_inline
    fn _insert_bit[start_bit: Int](self, val: Int64) -> Int64:
        return self.desc | (val << start_bit)

    @staticmethod
    fn create[
        stride_byte_offset: Int,
        leading_byte_offset: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](
        smem_ptr: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.SHARED
        ],
    ) -> Self:
        # TMA enumerates no swizzle, 32, 64, 128B as 0, 1, 2, 3.
        # WGMMA enumerates these as 0, 3, 2, 1.
        @parameter
        fn _convert_swizzle_enum[mode: Int32]() -> Int64:
            @parameter
            if mode == 0:
                return mode.cast[DType.int64]()
            else:
                return (4 - mode).cast[DType.int64]()

        alias swizzle = _convert_swizzle_enum[swizzle_mode._value]()
        var offset = Int64(0)
        var stride_dim = Int64(stride_byte_offset)
        var lead_dim = Int64(leading_byte_offset)

        # Extract 18 bits and ignore 4 LSB.
        var base_ptr = Int(smem_ptr)
        var start_address = (base_ptr & 0x3FFFF) >> 4

        # Start from LSB in case updated higher bits gets overwritten.
        var desc = Int64(0)
        # bits [48 .. 32]
        # bits  0:14 address in share memory
        desc = Self._insert_bit[0](desc, start_address)
        # bits 14:16 unused
        # bits 16:30 leading dim byte offset
        desc = Self._insert_bit[16](desc, lead_dim)
        # bits 30:32 unused
        # bits 32:46 stride dim byte offset
        desc = Self._insert_bit[32](desc, stride_dim)
        # bits 49:52 offset
        desc = Self._insert_bit[49](desc, offset)
        # bits 53:62 unused
        # bits 62:64 swizzle type
        desc = Self._insert_bit[62](desc, swizzle)

        return desc

    @always_inline
    fn __iadd__(mut self, offset: Int):
        self.desc += (offset & 0x3FFFF) >> 4

    @always_inline
    fn __add__(self, offset: Int) -> Self:
        return self.desc + ((offset & 0x3FFFF) >> 4)


@always_inline
fn wgmma_fence_aligned():
    __mlir_op.`nvvm.wgmma.fence.aligned`[_type=None]()


@always_inline
fn wgmma_commit_group_sync():
    __mlir_op.`nvvm.wgmma.commit.group.sync.aligned`[_type=None]()


@always_inline
fn wgmma_wait_group_sync():
    __mlir_op.`nvvm.wgmma.wait.group.sync.aligned`[
        _properties = __mlir_attr.`{group = 0 : i64}`, _type=None
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
    accum_type: DType = c_dtype,
    layout_a: StringLiteral = "row",
    layout_b: StringLiteral = "col",
](
    mat_a_desc: WGMMADescriptor,
    mat_b_desc: WGMMADescriptor,
    c_reg: SIMD[c_dtype, width],
) -> __type_of(c_reg):
    """Performs warp group async Matrix-multiply and accumulate(WGMMA) operation.
    """

    constrained[
        (m * n // 128) * sizeof[accum_type]() == width * sizeof[c_dtype](),
        "Number of output registers "
        + String(width)
        + " don't match the instruction shape "
        + String(Index(m, n, k)),
    ]()

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
    elif a_type is DType.float8_e4m3fn and b_type is DType.float8_e4m3fn:
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
    elif a_type is DType.float8_e5m2 and b_type is DType.float8_e5m2:
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
    elif a_type is DType.float8_e4m3fn and b_type is DType.float8_e5m2:
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
    elif a_type is DType.float8_e5m2 and b_type is DType.float8_e4m3fn:
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


@always_inline
fn wgmma_async[
    m: Int,
    n: Int,
    k: Int,
    a_dtype: DType,
    c_dtype: DType,
    frag_a_width: Int,
    frag_c_width: Int,
    /,
    *,
    a_type: DType,
    b_type: DType,
    accum_type: DType = c_dtype,
    layout_a: StringLiteral = "row",
    layout_b: StringLiteral = "col",
](
    mat_a_frag: SIMD[a_dtype, frag_a_width],
    mat_b_desc: WGMMADescriptor,
    c: SIMD[c_dtype, frag_c_width],
) -> __type_of(c):
    """Performs warp group async Matrix-multiply and accumulate(WGMMA) operation.
    """

    constrained[
        (m * n // 128) * sizeof[accum_type]()
        == frag_c_width * sizeof[c_dtype](),
        "Number of output registers "
        + String(frag_c_width)
        + " don't match the instruction shape "
        + String(Index(m, n, k)),
    ]()

    constrained[
        (m * k // 128) * sizeof[a_type]() == frag_a_width * sizeof[a_dtype](),
        "Number of input a registers "
        + String(frag_a_width)
        + " don't match the instruction shape "
        + String(Index(m, n, k)),
    ]()
    # for now, limited support
    constrained[m == 64]()
    constrained[k == 16]()
    constrained[a_type == DType.bfloat16]()
    constrained[b_type == DType.bfloat16]()
    constrained[accum_type == DType.float32]()
    constrained[c_dtype == DType.float32]()
    constrained[layout_a == "row"]()
    constrained[
        layout_b == "col" or (layout_b == "row" and b_type == DType.bfloat16)
    ]()

    var desc_b_value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i64](
        mat_b_desc.desc.value
    )
    alias scale_d = 1
    alias scale_a = 1
    alias scale_b = 1
    alias trans_b = layout_b == "row"

    @parameter
    if (
        m == 64
        and k == 16
        and a_type == b_type == DType.bfloat16
        and accum_type == c_dtype == DType.float32
    ):
        var a0 = bitcast[DType.uint32, 1](
            SIMD[DType.bfloat16, 2](
                rebind[Scalar[DType.bfloat16]](mat_a_frag[0]),
                rebind[Scalar[DType.bfloat16]](mat_a_frag[1]),
            )
        )
        var a1 = bitcast[DType.uint32, 1](
            SIMD[DType.bfloat16, 2](
                rebind[Scalar[DType.bfloat16]](mat_a_frag[2]),
                rebind[Scalar[DType.bfloat16]](mat_a_frag[3]),
            )
        )
        var a2 = bitcast[DType.uint32, 1](
            SIMD[DType.bfloat16, 2](
                rebind[Scalar[DType.bfloat16]](mat_a_frag[4]),
                rebind[Scalar[DType.bfloat16]](mat_a_frag[5]),
            )
        )
        var a3 = bitcast[DType.uint32, 1](
            SIMD[DType.bfloat16, 2](
                rebind[Scalar[DType.bfloat16]](mat_a_frag[6]),
                rebind[Scalar[DType.bfloat16]](mat_a_frag[7]),
            )
        )

        # fmt: off
        @parameter
        if n == 8:
            var r = inlined_assembly[
                """{
                    .reg .pred p;
                    setp.ne.b32 p, $9, 0;
                    wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 
                    {$0,   $1,   $2,   $3},   
                     {$4, $5, $6, $7},
                     $8, p, $10, $11, $12;
                    }""",
                _RegisterPackType[Float32, Float32, Float32, Float32],
                constraints = "=f," * 4 + "r,r,r,r,l,n,n,n,n,0,1,2,3",
            ](
                a0, a1, a2, a3,
                desc_b_value,
                scale_d, scale_a, scale_b, trans_b,
                c[0], c[1], c[2], c[3],
            )

            return rebind[__type_of(c)](
                SIMD[DType.float32, 4](r[0], r[1], r[2], r[3])
            )
        elif n == 16:
            var r = inlined_assembly[
                """{
                    .reg .pred p;
                    setp.ne.b32 p, $13, 0;
                    wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 
                    {$0,   $1,   $2,   $3,   $4,   $5,   $6,   $7},   
                     {$8, $9, $10, $11},
                     $12, p, $14, $15, $16;
                    }""",
                _RegisterPackType[
                    Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32,
                ],
                constraints = "=f," * 8 + "r,r,r,r,l,n,n,n,n,0,1,2,3,4,5,6,7",
            ](
                a0, a1, a2, a3,
                desc_b_value,
                scale_d, scale_a, scale_b, trans_b,
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            )

            return rebind[__type_of(c)](
                SIMD[DType.float32, 8](
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
                )
            )
        elif n == 32:
            var r = inlined_assembly[
                """{
                    .reg .pred p;
                    setp.ne.b32 p, $21, 0;
                    wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 
                    {$0,   $1,   $2,   $3,   $4,   $5,   $6,   $7,   
                     $8,   $9,   $10,  $11,  $12,  $13,  $14,  $15},  
                     {$16, $17, $18, $19},
                     $20, p, $22, $23, $24;
                    }""",
                _RegisterPackType[
                    Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32,
                ],
                constraints = "=f," * 16
                + "r,r,r,r,l,n,n,n,n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            ](
                a0, a1, a2, a3,
                desc_b_value,
                scale_d, scale_a, scale_b, trans_b,
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15],
            )

            return rebind[__type_of(c)](
                SIMD[DType.float32, 16](
                    r[0], r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],
                    r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15],
                )
            )
        elif n == 64:
            var r = inlined_assembly[
                """{
                    .reg .pred p;
                    setp.ne.b32 p, $37, 0;
                    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 
                    {$0,   $1,   $2,   $3,   $4,   $5,   $6,   $7,   
                     $8,   $9,   $10,  $11,  $12,  $13,  $14,  $15,  
                     $16,  $17,  $18,  $19,  $20,  $21,  $22,  $23,  
                     $24,  $25,  $26,  $27,  $28,  $29,  $30,  $31},  
                     {$32, $33, $34, $35},
                     $36, p, $38, $39, $40;
                    }""",
                _RegisterPackType[
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32
                ],
                constraints = "=f," * 32
                + "r,r,r,r,l,n,n,n,n,"
                + "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
                + "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
            ](
                a0, a1, a2, a3,
                desc_b_value,
                scale_d, scale_a, scale_b, trans_b,
                c[0],  c[1],  c[2],  c[3],  c[4],  c[5],  c[6],  c[7],
                c[8],  c[9],  c[10], c[11], c[12], c[13], c[14], c[15],
                c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23],
                c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31],
            )

            return rebind[__type_of(c)](
                SIMD[DType.float32, 32](
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                    r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15],
                    r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23],
                    r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31],
                )
            )
        elif n == 128:
            var r = inlined_assembly[
                """{
                    .reg .pred p;
                    setp.ne.b32 p, $69, 0;
                    wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 
                    {$0,   $1,   $2,   $3,   $4,   $5,   $6,   $7,   
                     $8,   $9,   $10,  $11,  $12,  $13,  $14,  $15,  
                     $16,  $17,  $18,  $19,  $20,  $21,  $22,  $23,  
                     $24,  $25,  $26,  $27,  $28,  $29,  $30,  $31,  
                     $32,  $33,  $34,  $35,  $36,  $37,  $38,  $39,  
                     $40,  $41,  $42,  $43,  $44,  $45,  $46,  $47,  
                     $48,  $49,  $50,  $51,  $52,  $53,  $54,  $55,  
                     $56,  $57,  $58,  $59,  $60,  $61,  $62,  $63},
                     {$64, $65, $66, $67},
                     $68, p, $70, $71, $72;
                    }""",
                _RegisterPackType[
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32
                ],
                constraints = "=f," * 64
                + "r,r,r,r,l,n,n,n,n,"
                + "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
                + "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,"
                + "32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,"
                + "48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63",
            ](
                a0, a1, a2, a3,
                desc_b_value,
                scale_d, scale_a, scale_b, trans_b,
                c[0],  c[1],  c[2],  c[3],  c[4],  c[5],  c[6],  c[7],
                c[8],  c[9],  c[10], c[11], c[12], c[13], c[14], c[15],
                c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23],
                c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31],
                c[32], c[33], c[34], c[35], c[36], c[37], c[38], c[39],
                c[40], c[41], c[42], c[43], c[44], c[45], c[46], c[47],
                c[48], c[49], c[50], c[51], c[52], c[53], c[54], c[55],
                c[56], c[57], c[58], c[59], c[60], c[61], c[62], c[63],
            )
            return rebind[__type_of(c)](
                SIMD[DType.float32, 64](
                    r[0],  r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],
                    r[8],  r[9],  r[10], r[11], r[12], r[13], r[14], r[15],
                    r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23],
                    r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31],
                    r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39],
                    r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47],
                    r[48], r[49], r[50], r[51], r[52], r[53], r[54], r[55],
                    r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                )
            )
        elif n == 256:
            var r = inlined_assembly[
                """{
                    .reg .pred p;
                    setp.ne.b32 p, $133, 0;
                    wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 
                    {$0,   $1,   $2,   $3,   $4,   $5,   $6,   $7,   
                     $8,   $9,   $10,  $11,  $12,  $13,  $14,  $15,  
                     $16,  $17,  $18,  $19,  $20,  $21,  $22,  $23,  
                     $24,  $25,  $26,  $27,  $28,  $29,  $30,  $31,  
                     $32,  $33,  $34,  $35,  $36,  $37,  $38,  $39,  
                     $40,  $41,  $42,  $43,  $44,  $45,  $46,  $47,  
                     $48,  $49,  $50,  $51,  $52,  $53,  $54,  $55,  
                     $56,  $57,  $58,  $59,  $60,  $61,  $62,  $63,  
                     $64,  $65,  $66,  $67,  $68,  $69,  $70,  $71,  
                     $72,  $73,  $74,  $75,  $76,  $77,  $78,  $79,  
                     $80,  $81,  $82,  $83,  $84,  $85,  $86,  $87,  
                     $88,  $89,  $90,  $91,  $92,  $93,  $94,  $95,  
                     $96,  $97,  $98,  $99,  $100, $101, $102, $103, 
                     $104, $105, $106, $107, $108, $109, $110, $111, 
                     $112, $113, $114, $115, $116, $117, $118, $119, 
                     $120, $121, $122, $123, $124, $125, $126, $127}, 
                     {$128, $129, $130, $131},
                     $132, p, $134, $135, $136;
                    }""",
                _RegisterPackType[
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32, 
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32, Float32, Float32, Float32, Float32,
                    Float32, Float32,
                ],
                constraints = "=f," * 128
                + "r,r,r,r,l,n,n,n,n,"
                + "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,"
                + "20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,"
                + "37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,"
                + "54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,"
                + "71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,"
                + "88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,"
                + "104,105,106,107,108,109,110,111,112,113,114,115,"
                + "116,117,118,119,120,121,122,123,124,125,126,127",
            ](
                a0, a1, a2, a3,
                desc_b_value,
                scale_d, scale_a, scale_b, trans_b,
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
                c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18], 
                c[19], c[20], c[21], c[22], c[23], c[24], c[25], c[26], c[27], 
                c[28], c[29], c[30], c[31], c[32], c[33], c[34], c[35], c[36], 
                c[37], c[38], c[39], c[40], c[41], c[42], c[43], c[44], c[45], 
                c[46], c[47], c[48], c[49], c[50], c[51], c[52], c[53], c[54], 
                c[55], c[56], c[57], c[58], c[59], c[60], c[61], c[62], c[63], 
                c[64], c[65], c[66], c[67], c[68], c[69], c[70], c[71], c[72], 
                c[73], c[74], c[75], c[76], c[77], c[78], c[79], c[80], c[81], 
                c[82], c[83], c[84], c[85], c[86], c[87], c[88], c[89], c[90],
                c[91], c[92], c[93], c[94], c[95], c[96], c[97], c[98], c[99],
                c[100], c[101], c[102], c[103], c[104], c[105], c[106], c[107],
                c[108], c[109], c[110], c[111], c[112], c[113], c[114], c[115],
                c[116], c[117], c[118], c[119], c[120], c[121], c[122], c[123],
                c[124], c[125], c[126], c[127],
            )

            return rebind[__type_of(c)](
                SIMD[DType.float32, 128](
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                    r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
                    r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
                    r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36],
                    r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45],
                    r[46], r[47], r[48], r[49], r[50], r[51], r[52], r[53], r[54],
                    r[55], r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                    r[64], r[65], r[66], r[67], r[68], r[69], r[70], r[71], r[72],
                    r[73], r[74], r[75], r[76], r[77], r[78], r[79], r[80], r[81],
                    r[82], r[83], r[84], r[85], r[86], r[87], r[88], r[89], r[90],
                    r[91], r[92], r[93], r[94], r[95], r[96], r[97], r[98], r[99],
                    r[100], r[101], r[102], r[103], r[104], r[105], r[106], r[107],
                    r[108], r[109], r[110], r[111], r[112], r[113], r[114], r[115],
                    r[116], r[117], r[118], r[119], r[120], r[121], r[122], r[123],
                    r[124], r[125], r[126], r[127],
                )
            )
        else:
            constrained[False, "n is invalid"]()
            return c
        # fmt: on

    else:
        constrained[False, "unsupported config"]()
        return c
