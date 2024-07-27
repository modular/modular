# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.memory import AddressSpace, async_copy
from gpu.sync import mbarrier, mbarrier_init, mbarrier_test_wait
from memory import stack_allocation
from memory import UnsafePointer
from testing import *


@always_inline
fn _get_nvptx_target_sm90() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ]


fn test_mbarrier(
    addr0: UnsafePointer[Int8],
    addr1: UnsafePointer[UInt8],
    addr2: UnsafePointer[Float32, AddressSpace.GLOBAL],
    addr3: UnsafePointer[Float32, AddressSpace.SHARED],
    addr4: UnsafePointer[Float64, AddressSpace.GLOBAL],
    addr5: UnsafePointer[Float64, AddressSpace.SHARED],
):
    mbarrier(addr0)
    mbarrier(addr1)
    mbarrier(addr2)
    mbarrier(addr3)
    mbarrier(addr4)
    mbarrier(addr5)


@always_inline
fn _verify_mbarrier(asm: String) raises -> None:
    assert_true("cp.async.mbarrier.arrive.b64" in asm)
    assert_true("cp.async.mbarrier.arrive.shared.b64" in asm)


def test_mbarrier_sm80():
    alias asm = str(
        _compile_code[test_mbarrier, target = _get_nvptx_target()]().asm
    )
    _verify_mbarrier(asm)


def test_mbarrier_sm90():
    alias asm = str(
        _compile_code[test_mbarrier, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_mbarrier(asm)


fn test_mbarrier_init(
    shared_mem: UnsafePointer[Int32, AddressSpace.SHARED],
):
    mbarrier_init(shared_mem, 4)


@always_inline
fn _verify_mbarrier_init(asm: String) raises -> None:
    assert_true("ld.param.u64" in asm)
    assert_true("mov.b32" in asm)
    assert_true("mbarrier.init.shared.b64" in asm)


def test_mbarrier_init_sm80():
    alias asm = str(
        _compile_code[test_mbarrier_init, target = _get_nvptx_target()]().asm
    )
    _verify_mbarrier_init(asm)


def test_mbarrier_init_sm90():
    alias asm = str(
        _compile_code[
            test_mbarrier_init, target = _get_nvptx_target_sm90()
        ]().asm
    )
    _verify_mbarrier_init(asm)


fn test_mbarrier_test_wait(
    shared_mem: UnsafePointer[Int32, AddressSpace.SHARED], state: Int
):
    var done = False
    while not done:
        done = mbarrier_test_wait(shared_mem, state)


@always_inline
fn _verify_mbarrier_test_wait(asm: String) raises -> None:
    assert_true("mbarrier.test_wait.shared.b64" in asm)


def test_mbarrier_test_wait_sm80():
    alias asm = str(
        _compile_code[
            test_mbarrier_test_wait, target = _get_nvptx_target()
        ]().asm
    )
    _verify_mbarrier_test_wait(asm)


def test_mbarrier_test_wait_sm90():
    alias asm = str(
        _compile_code[
            test_mbarrier_test_wait, target = _get_nvptx_target_sm90()
        ]().asm
    )
    assert_true("mbarrier.test_wait.shared.b64" in asm)


fn test_async_copy(src: UnsafePointer[Float32, AddressSpace.GLOBAL]):
    var barrier = stack_allocation[
        sizeof[DType.int32](), DType.int32, address_space = AddressSpace.SHARED
    ]()
    var shared_mem = stack_allocation[
        4, DType.float32, address_space = AddressSpace.SHARED
    ]()
    async_copy[4](src, shared_mem)
    async_copy[16](src, shared_mem)


@always_inline
fn _verify_async_copy(asm: String) raises -> None:
    assert_true("cp.async.ca.shared.global" in asm)
    assert_true("cp.async.cg.shared.global" in asm)


def test_async_copy_sm80():
    alias asm = str(
        _compile_code[test_async_copy, target = _get_nvptx_target()]().asm
    )
    _verify_async_copy(asm)


def test_async_copy_sm90():
    alias asm = str(
        _compile_code[test_async_copy, target = _get_nvptx_target_sm90()]().asm
    )
    _verify_async_copy(asm)


def main():
    test_mbarrier_sm80()
    test_mbarrier_sm90()
    test_mbarrier_init_sm80()
    test_mbarrier_init_sm90()
    test_mbarrier_test_wait_sm80()
    test_mbarrier_test_wait_sm90()
    test_async_copy_sm80()
    test_async_copy_sm90()
