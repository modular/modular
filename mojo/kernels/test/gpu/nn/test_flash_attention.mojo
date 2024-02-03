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
from runtime.llcl import Runtime
from utils.index import Index
from utils.list import DimList
from BatchedMatmul import batched_matmul
from NN.Softmax import softmax
from gpu.host.event import time_function
from sys import argv

from gpu import *
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)

from NN.MultiHeadAttention import (
    flash_attention_kernel,
    flash_attention_kernel_flexible_seqlen,
    _naive_attention_with_transpose,
)

alias type = DType.float32


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# CHECK-LABEL: test_flash_attention
fn test(seq_len: Int, num_keys: Int, is_benchmark: Bool = False) raises:
    print("test_flash_attention")

    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 32
    alias depth = 128
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    # Q, K, V shapes.
    let q_size = batch_size * num_heads * seq_len * depth
    let k_size = batch_size * num_heads * num_keys * depth
    let v_size = k_size
    let o_size = q_size

    # Allocate memory for all variables.
    let q_ptr = DTypePointer[type].alloc(q_size)
    let k_ptr = DTypePointer[type].alloc(k_size)
    let v_ptr = DTypePointer[type].alloc(v_size)
    let mask_ptr = DTypePointer[type].alloc(seq_len * num_keys)
    let output_ptr = DTypePointer[type].alloc(o_size)
    let flash_output_ptr = DTypePointer[type].alloc(o_size)

    # Q, K, V are randomly initalized.
    rand[type](q_ptr, q_size)
    rand[type](k_ptr, k_size)
    rand[type](v_ptr, v_size)
    rand[type](mask_ptr, seq_len * num_keys)

    # Contruct buffers.
    let q = NDBuffer[type, 4, DimList.create_unknown[4]()](
        q_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    let k = NDBuffer[type, 4, DimList.create_unknown[4]()](
        k_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    let v = NDBuffer[type, 4, DimList.create_unknown[4]()](
        v_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    let mask = NDBuffer[type, 2, DimList.create_unknown[2]()](
        mask_ptr, Index(seq_len, num_keys)
    )
    let output = NDBuffer[type, 4, DimList.create_unknown[4]()](
        output_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    _naive_attention_with_transpose[type](
        rebind[NDBuffer[type, 4, DimList.create_unknown[4]()]](output),
        rebind[NDBuffer[type, 4, DimList.create_unknown[4]()]](q),
        rebind[NDBuffer[type, 4, DimList.create_unknown[4]()]](k),
        rebind[NDBuffer[type, 4, DimList.create_unknown[4]()]](v),
        rebind[NDBuffer[type, 2, DimList.create_unknown[2]()]](mask),
        scale,
    )

    let stream = Stream()

    # Device pointers
    let q_device_ptr = _malloc[type](q_size)
    let k_device_ptr = _malloc[type](k_size)
    let v_device_ptr = _malloc[type](v_size)
    let mask_device_ptr = _malloc[type](seq_len * num_keys)
    let output_device_ptr = _malloc[type](o_size)

    # Copy from host to device
    _copy_host_to_device(q_device_ptr, q_ptr, q_size)
    _copy_host_to_device(k_device_ptr, k_ptr, k_size)
    _copy_host_to_device(v_device_ptr, v_ptr, v_size)
    _copy_host_to_device(mask_device_ptr, mask_ptr, seq_len * num_keys)

    alias q_tile_num_rows = 32

    if seq_len == num_keys and seq_len % 128 == 0:
        let func = Function[
            fn (
                DTypePointer[type],
                DTypePointer[type],
                DTypePointer[type],
                DTypePointer[type],
                DTypePointer[type],
                Float32,
                Scalar[DType.uint32],
                Scalar[DType.uint32],
            ) -> None, flash_attention_kernel[
                BM=32,  # q_tile_num_rows,
                BN=128,  # kv_tile_num_rows,
                BK=16,
                depth=128,
                num_heads=num_heads,
                TM=8,
                TN=4,
                num_threads=128,  # q_tile_num_rows * kv_tile_num_rows,
            ]
        ]()

        if is_benchmark:
            alias nrun = 1000

            @always_inline
            @parameter
            fn run_func(stream: Stream) raises:
                for i in range(nrun):
                    func(
                        stream,
                        # grid
                        (
                            div_ceil(seq_len, q_tile_num_rows),
                            num_heads,
                            batch_size,
                        ),
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
                    )

            # Warmup
            run_func(stream)

            var nstime = time_function[run_func](stream) / nrun
            let sectime = nstime / 1000000
            print(nrun, "runs avg", sectime, "ms")

        else:
            func(
                stream,
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
            )

    else:
        let func = Function[
            fn (
                DTypePointer[type],
                DTypePointer[type],
                DTypePointer[type],
                DTypePointer[type],
                DTypePointer[type],
                Float32,
                Scalar[DType.uint32],
                Scalar[DType.uint32],
                Scalar[DType.uint32],
            ) -> None, flash_attention_kernel_flexible_seqlen[
                BM=32,  # q_tile_num_rows,
                BN=128,  # kv_tile_num_rows,
                BK=16,
                depth=128,
                num_heads=num_heads,
                TM=8,
                TN=4,
                num_threads=128,  # q_tile_num_rows * kv_tile_num_rows,
            ]
        ]()

        func(
            stream,
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
            num_keys,
        )

    synchronize()

    _copy_device_to_host(flash_output_ptr, output_device_ptr, q_size)

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

    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            # Context encoding demo.
            test(100, 100)
            test(31, 31)
            test(1024, 1024, is_benchmark())  # only benchmark a large shape
            # Token generation demo.
            test(1, 1)
            test(1, 2)
            test(1, 3)
            test(1, 4)
            test(1, 5)
            test(1, 6)
            test(1, 7)
            test(1, 8)
            test(1, 9)
            test(1, 10)
            test(1, 11)
            test(1, 12)
            test(1, 13)
            test(1, 14)
            test(1, 15)
            test(1, 20)
            test(1, 25)
            test(1, 30)
            test(1, 50)
            test(1, 100)
            test(1, 200)
    except e:
        print("CUDA_ERROR:", e)
