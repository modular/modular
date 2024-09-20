# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from collections import OptionalReg
from memory import UnsafePointer

from compile import Info
from gpu.host._compile import _compile_code
from gpu.memory import CacheEviction, CacheOperation, load
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


# Get the asm field from a compile Info object at compile time.
def comptime_asm[info: Info]() -> String:
    alias asm = info.asm
    return asm


def test_load():
    assert_true(
        "ld.global "
        in comptime_asm[
            _compile_code[
                load_value[width=1, prefetch_size=128], emission_kind="ptx"
            ]()
        ]()
    )

    assert_true(
        "ld.global.L2::128B.v2.u32 "
        in comptime_asm[
            _compile_code[
                load_value[width=2, prefetch_size=128], emission_kind="ptx"
            ]()
        ]()
    )

    assert_true(
        "ld.global.L2::128B.v4.u32 "
        in comptime_asm[
            _compile_code[
                load_value[width=4, prefetch_size=128], emission_kind="ptx"
            ]()
        ]()
    )

    assert_true(
        "ld.global.L2::256B.v4.u32 "
        in comptime_asm[
            _compile_code[
                load_value[width=4, prefetch_size=256], emission_kind="ptx"
            ]()
        ]()
    )

    assert_equal(
        comptime_asm[
            _compile_code[
                load_value[width=64, prefetch_size=128], emission_kind="ptx"
            ]()
        ]().count("ld.global.L2::128B.v4.u32 "),
        16,
    )

    assert_true(
        "ld.global.lu.v2.u32 "
        in comptime_asm[
            _compile_code[
                load_value[
                    type = DType.uint32,
                    width=2,
                    prefetch_size=None,
                    cache_policy = CacheOperation.LAST_USE,
                ],
                emission_kind="ptx",
            ]()
        ]()
    )

    assert_true(
        "ld.global.nc.v2.u32 "
        in comptime_asm[
            _compile_code[
                load_value[type = DType.uint32, width=2, read_only=True],
                emission_kind="ptx",
            ]()
        ]()
    )


def main():
    test_load()
