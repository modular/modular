# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, min, abs, rsqrt
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer
from random import rand
from runtime.llcl import OwningOutputChainPtr, OutputChainPtr, Runtime
from utils.index import Index
from utils.list import DimList
from BatchedMatmul import batched_matmul
from Softmax import softmax

from gpu import *
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)

from MultiHeadAttention import (
    flash_attention_kernel,
    _naive_attention_with_transpose,
)

alias type = DType.float32


# CHECK-LABEL: test_flash_attention
fn test() raises:
    print("test_flash_attention")

    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 32
    alias seq_len = 1024
    alias depth = 128
    alias mask_val = Float32(-1e10)
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    # Q, K, V shapes.
    alias BSHD = DimList(batch_size, seq_len, num_heads, depth)
    alias BHSD = DimList(batch_size, num_heads, seq_len, depth)
    alias BHDS = DimList(batch_size, num_heads, depth, seq_len)

    alias qkv_size = batch_size * num_heads * seq_len * depth

    # Allocate memory for all variables.
    let q_ptr = DTypePointer[type].alloc(qkv_size)
    let k_ptr = DTypePointer[type].alloc(qkv_size)
    let v_ptr = DTypePointer[type].alloc(qkv_size)
    let mask_ptr = DTypePointer[type].alloc(seq_len * seq_len)
    let output_ptr = DTypePointer[type].alloc(qkv_size)
    let flash_output_ptr = DTypePointer[type].alloc(qkv_size)

    # Q, K, V are randomly initalized.
    rand[type](q_ptr, qkv_size)
    rand[type](k_ptr, qkv_size)
    rand[type](v_ptr, qkv_size)

    # Mask is set for half of the sequence.
    for b in range(seq_len):
        for i in range(seq_len // 2):
            mask_ptr.offset(b * seq_len + i).store(0.0)
        for i in range(seq_len // 2, seq_len):
            mask_ptr.offset(b * seq_len + i).store(mask_val)

    # Contruct buffers.
    let q = NDBuffer[4, BSHD, type](q_ptr)
    let k = NDBuffer[4, BSHD, type](k_ptr)
    let v = NDBuffer[4, BSHD, type](v_ptr)
    let mask = NDBuffer[2, DimList.create_unknown[2](), type](
        mask_ptr, Index(seq_len, seq_len)
    )
    let output = NDBuffer[4, BSHD, type](output_ptr)

    _naive_attention_with_transpose[type, BSHD, BHSD, BHDS](
        output, q, k, v, mask, scale
    )

    let stream = Stream()

    # Device pointers
    let q_device_ptr = _malloc[type](qkv_size)
    let k_device_ptr = _malloc[type](qkv_size)
    let v_device_ptr = _malloc[type](qkv_size)
    let mask_device_ptr = _malloc[type](seq_len * seq_len)
    let output_device_ptr = _malloc[type](qkv_size)

    # Copy from host to device
    _copy_host_to_device(q_device_ptr, q_ptr, qkv_size)
    _copy_host_to_device(k_device_ptr, k_ptr, qkv_size)
    _copy_host_to_device(v_device_ptr, v_ptr, qkv_size)
    _copy_host_to_device(mask_device_ptr, mask_ptr, seq_len * seq_len)

    alias q_tile_num_rows = 32
    alias kv_tile_num_rows = WARP_SIZE

    let func = Function[
        fn (
            DTypePointer[type],
            DTypePointer[type],
            DTypePointer[type],
            DTypePointer[type],
            DTypePointer[type],
            Float32,
            Int,
            Int,
        ) -> None, flash_attention_kernel[
            BM=32,  # q_tile_num_rows,
            BN=128,  # kv_tile_num_rows,
            BK=16,
            depth=128,
            num_heads=32,
            TM=8,
            TN=4,
            num_threads=128,  # q_tile_num_rows * kv_tile_num_rows,
        ]
    ]()

    func(
        # grid
        (div_ceil(seq_len, q_tile_num_rows), num_heads, batch_size),
        # block
        (128, 1, 1),
        q_device_ptr,
        k_device_ptr,
        v_device_ptr,
        mask_device_ptr,
        output_device_ptr,
        scale,
        batch_size,
        seq_len,
        stream=stream,
    )

    synchronize()

    _copy_device_to_host(flash_output_ptr, output_device_ptr, qkv_size)

    var succeed = True
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                let expect = output_ptr.load(d + depth * (h + s * num_heads))
                let actual = flash_output_ptr.load(
                    d + depth * (h + s * num_heads)
                )
                if abs(expect - actual) > 1e-4 * abs(expect):
                    print(d, expect, actual)
                    succeed = False
                    break

    _free(q_device_ptr)
    _free(k_device_ptr)
    _free(v_device_ptr)
    _free(mask_device_ptr)
    _free(output_device_ptr)

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()

    # CHECK: Succeed
    if succeed:
        print("Succeed")

    # _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            test()
    except e:
        print("CUDA_ERROR:", e)
