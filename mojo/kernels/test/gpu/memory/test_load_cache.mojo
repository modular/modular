# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug %s

from collections import OptionalReg

from compile import Info
from gpu.host._compile import _compile_code_asm
from gpu.memory import CacheEviction, CacheOperation, load
from memory import UnsafePointer
from testing import assert_equal, assert_true


fn load_value[
    *,
    type: DType = DType.uint32,
    width: Int = 1,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](ptr: UnsafePointer[Scalar[type]]) -> SIMD[type, width]:
    return load[
        width=width,
        read_only=read_only,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
    ](ptr)


def test_load():
    assert_true(
        "ld.global "
        in _compile_code_asm[
            load_value[width=1, prefetch_size=128], emission_kind="asm"
        ]()
    )

    assert_true(
        "ld.global.L2::128B.v2.u32 "
        in _compile_code_asm[
            load_value[width=2, prefetch_size=128], emission_kind="asm"
        ]()
    )

    assert_true(
        "ld.global.L2::128B.v4.u32 "
        in _compile_code_asm[
            load_value[width=4, prefetch_size=128], emission_kind="asm"
        ]()
    )

    assert_true(
        "ld.global.L2::256B.v4.u32 "
        in _compile_code_asm[
            load_value[width=4, prefetch_size=256], emission_kind="asm"
        ]()
    )

    assert_equal(
        String(
            _compile_code_asm[
                load_value[width=64, prefetch_size=128], emission_kind="asm"
            ]()
        ).count("ld.global.L2::128B.v4.u32 "),
        16,
    )

    assert_true(
        "ld.global.lu.v2.u32 "
        in _compile_code_asm[
            load_value[
                type = DType.uint32,
                width=2,
                prefetch_size=None,
                cache_policy = CacheOperation.LAST_USE,
            ],
            emission_kind="asm",
        ]()
    )

    assert_true(
        "ld.global.nc.v2.u32 "
        in _compile_code_asm[
            load_value[type = DType.uint32, width=2, read_only=True],
            emission_kind="asm",
        ]()
    )


def main():
    test_load()
