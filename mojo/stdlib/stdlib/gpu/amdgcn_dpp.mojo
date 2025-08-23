from gpu import *
from gpu.host.info import _get_info_from_target, Vendor
from gpu.warp import *
from sys import llvm_intrinsic, is_amd_gpu
from sys.info import _accelerator_arch

# Instructions used in this module can be found by looking for "DPP_CTRL" in these documents describing the ISA:
#
# [1]: Vega https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-shader-instruction-set-architecture.pdf
# [2]: MI300 https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf

# TODO: put somewhere more appropriate
alias current_target = _get_info_from_target[_accelerator_arch()]()

fn _gpu_arch_check[f: fn(Float32)->Bool]() -> Bool:
    return current_target.vendor == Vendor.AMD_GPU and f(current_target.compute)

fn _amdgcn_dpp[
    dtype: DType,
    width: Int, //,
    dpp_ctrl: UInt32,
    row_mask: UInt32 = 0b1111,
    bank_mask: UInt32 = 0b1111,
](old: SIMD[dtype, width], src: SIMD[dtype, width]) -> SIMD[dtype, width]:
    constrained[is_amd_gpu()]()
    constrained[
        dtype.bitwidth() in (32, 64), "Can only use DPP with 32/64-bit dtypes"
    ]()

    bound_ctrl = True
    return llvm_intrinsic["llvm.amdgcn.update.dpp", SIMD[dtype, width]](
        old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl
    )


fn amdgcn_row_mirror[
    dtype: DType, width: Int, //
](old: SIMD[dtype, width], src: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return _amdgcn_dpp[dpp_ctrl=0x140](old, src)


fn amdgcn_row_shift_left[
    dtype: DType, width: Int, //, offset: Int
](src: SIMD[dtype, width], old: SIMD[dtype, width] = 0) -> SIMD[dtype, width]:
    constrained[
        offset > 0 and offset < 16, "Can only shift row by up to 15 positions"
    ]()
    return _amdgcn_dpp[dpp_ctrl = 0x100 + offset](old, src)


fn amdgcn_row_rotate_left[
    dtype: DType, width: Int, //, offset: Int
](src: SIMD[dtype, width], old: SIMD[dtype, width] = 0) -> SIMD[dtype, width]:
    constrained[
        offset > 0 and offset < 16, "Can only rotate row by up to 15 positions"
    ]()
    return _amdgcn_dpp[dpp_ctrl = 0x110 + offset](old, src)

# Seen in Vega ISA[1] up to MI300 ISA[2]
fn amdgcn_supports_shifts() -> Bool:
    fn check(version: Float32) -> Bool:
      return version >= 9.0 and version < 10.0
    return _gpu_arch_check[check]()

fn amdgcn_shift_left[
    dtype: DType, width: Int, //
](src: SIMD[dtype, width], old: SIMD[dtype, width] = 0) -> SIMD[dtype, width]:
    constrained[amdgcn_supports_shifts(), "DPP wavefront shift only supported on CDNA"]()
    return _amdgcn_dpp[dpp_ctrl=0x130](old, src)


fn amdgcn_rotate_left[
    dtype: DType, width: Int, //
](src: SIMD[dtype, width], old: SIMD[dtype, width] = 0) -> SIMD[dtype, width]:
    return _amdgcn_dpp[dpp_ctrl=0x134](old, src)


fn amdgcn_row_read_lane[
    dtype: DType, width: Int, //, offset: Int
](src: SIMD[dtype, width], old: SIMD[dtype, width] = 0) -> SIMD[dtype, width]:
    constrained[
        offset >= 0 and offset < 16, "Can only broadcast within each row (0-15)"
    ]()
    return _amdgcn_dpp[dpp_ctrl = 0x150 + offset](old, src)


fn amdgcn_quad_perm[
    dtype: DType, width: Int, //, perm: Int
](src: SIMD[dtype, width], old: SIMD[dtype, width] = 0) -> SIMD[dtype, width]:
    constrained[
        perm >= 0 and perm <= 0xFF, "DPP_QUAD_PERM must be between 0 and 0xFF"
    ]()
    return _amdgcn_dpp[dpp_ctrl = 0x0 + perm](old, src)


fn amdgcn_quad_shuffle_xor[
    dtype: DType, width: Int, //, mask: Int
](src: SIMD[dtype, width], old: SIMD[dtype, width] = 0) -> SIMD[dtype, width]:
    constrained[
        mask >= 0 and mask <= 3, "Quad shuffle mask must be between 0 and 3"
    ]()

    fn calculate_bitmask(xor: Int) -> Int:
        # calculate lane indices for a quad
        mask = 0
        for i in range(4):
            mask |= (i ^ xor) << 2 * i
        return mask

    alias bitmask = calculate_bitmask(mask)
    return amdgcn_quad_perm[bitmask](src, old)

