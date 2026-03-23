# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Fused cross-entropy loss kernels for training.

Provides compiler-registered operations for cross-entropy loss:
- CrossEntropyFwd: Fused log-softmax + NLL forward pass producing scalar loss.
- CrossEntropyBwd: Fused softmax - one_hot backward pass producing grad_logits.
"""

from std.math import exp, log

import compiler_internal as compiler
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.primitives import block
from std.os.atomic import Atomic
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList
from std.utils.numerics import get_accum_type
from tensor import InputTensor, OutputTensor


# ===----------------------------------------------------------------------=== #
# Forward kernel: fused log-softmax + NLL loss
# ===----------------------------------------------------------------------=== #


def _cross_entropy_fwd_gpu[
    BLOCK_SIZE: Int,
    logits_dtype: DType,
    targets_dtype: DType,
    output_dtype: DType,
    LogitsLayoutType: layout.TensorLayout,
    TargetsLayoutType: layout.TensorLayout,
    OutputLayoutType: layout.TensorLayout,
    logits_origin: ImmutOrigin,
    targets_origin: ImmutOrigin,
    output_origin: MutOrigin,
    accum_type: DType = get_accum_type[logits_dtype](),
](
    logits: layout.TileTensor[
        logits_dtype, LogitsLayoutType, logits_origin
    ],
    targets: layout.TileTensor[
        targets_dtype, TargetsLayoutType, targets_origin
    ],
    output: layout.TileTensor[output_dtype, OutputLayoutType, output_origin],
    batch_size: Int,
    vocab_size: Int,
    ignore_index: Int,
    inv_count: Scalar[DType.float32],
):
    """GPU kernel for cross-entropy forward pass.

    Each block processes one row. Computes numerically stable log-softmax at the
    target index, negates it, scales by inv_count, and atomically accumulates
    into the scalar output.
    """
    var tid = Int(thread_idx.x)

    for row_idx in range(Int(block_idx.x), batch_size, Int(grid_dim.x)):
        var target = Int(targets.ptr[row_idx])

        if target == ignore_index:
            continue

        # Pass 1: row max for numerical stability.
        var row_max = Scalar[accum_type].MIN
        for col in range(tid, vocab_size, BLOCK_SIZE):
            var val = logits.ptr[row_idx * vocab_size + col].cast[accum_type]()
            row_max = max(row_max, val)

        row_max = block.max[block_size=BLOCK_SIZE](row_max)

        # Pass 2: sum(exp(logit - max)).
        var exp_sum = Scalar[accum_type](0)
        for col in range(tid, vocab_size, BLOCK_SIZE):
            var val = logits.ptr[row_idx * vocab_size + col].cast[accum_type]()
            exp_sum += exp(val - row_max)

        exp_sum = block.sum[block_size=BLOCK_SIZE](exp_sum)

        # Thread 0 computes the scaled loss and atomically accumulates it.
        if tid == 0:
            var target_logit = logits.ptr[
                row_idx * vocab_size + target
            ].cast[accum_type]()
            var log_softmax = target_logit - row_max - log(exp_sum)
            var row_loss = -log_softmax * inv_count.cast[accum_type]()

            _ = Atomic.fetch_add(
                output.ptr.bitcast[Scalar[output_dtype]](),
                row_loss.cast[output_dtype](),
            )


def _cross_entropy_fwd_cpu[
    logits_dtype: DType,
    targets_dtype: DType,
    output_dtype: DType,
](
    logits_ptr: UnsafePointer[Scalar[logits_dtype]],
    targets_ptr: UnsafePointer[Scalar[targets_dtype]],
    output_ptr: UnsafePointer[Scalar[output_dtype]],
    batch_size: Int,
    vocab_size: Int,
    ignore_index: Int,
):
    """CPU fallback for cross-entropy forward pass."""
    comptime accum_type = get_accum_type[logits_dtype]()
    var total_loss = Scalar[accum_type](0)
    var count = 0

    for row in range(batch_size):
        var target = Int(targets_ptr[row])
        if target == ignore_index:
            continue

        var row_max = Scalar[accum_type].MIN
        for col in range(vocab_size):
            var val = logits_ptr[row * vocab_size + col].cast[accum_type]()
            row_max = max(row_max, val)

        var exp_sum = Scalar[accum_type](0)
        for col in range(vocab_size):
            var val = logits_ptr[row * vocab_size + col].cast[accum_type]()
            exp_sum += exp(val - row_max)

        var target_logit = logits_ptr[
            row * vocab_size + target
        ].cast[accum_type]()
        var log_softmax = target_logit - row_max - log(exp_sum)
        total_loss += -log_softmax
        count += 1

    if count > 0:
        output_ptr[0] = (total_loss / Scalar[accum_type](count)).cast[
            output_dtype
        ]()
    else:
        output_ptr[0] = Scalar[output_dtype](0)


@compiler.register("training.cross_entropy_fwd")
struct CrossEntropyFwd:
    """Fused log-softmax + NLL forward pass for cross-entropy loss.

    Computes the mean cross-entropy loss over a batch, with support for
    ignoring specific target indices (e.g., padding tokens).

    Tensor Shapes:
        - logits: (batch, vocab) - Unnormalized log-probabilities.
        - targets: (batch,) - Ground-truth class indices.
        - ignore_index_tensor: (1,) - Index value to ignore in loss.
        - output: (1,) - Scalar mean loss (float32).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=1, ...],
        logits: InputTensor[dtype=dtype, rank=2, ...],
        targets: InputTensor[dtype=DType.int64, rank=1, ...],
        ignore_index_tensor: InputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var batch_size = logits.dim_size(0)
        var vocab_size = logits.dim_size(1)
        var ignore_index = Int(
            ignore_index_tensor.to_tile_tensor[DType.int32]().ptr[0]
        )

        # Zero-initialize the output before atomic accumulation.
        output.to_tile_tensor[DType.int32]().ptr.bitcast[
            Scalar[DType.float32]
        ]()[0] = Scalar[DType.float32](0)

        comptime if is_cpu[target]():
            _cross_entropy_fwd_cpu[dtype, DType.int64, DType.float32](
                logits.to_tile_tensor[DType.int32]().ptr.bitcast[
                    Scalar[dtype]
                ](),
                targets.to_tile_tensor[DType.int32]().ptr.bitcast[
                    Scalar[DType.int64]
                ](),
                output.to_tile_tensor[DType.int32]().ptr.bitcast[
                    Scalar[DType.float32]
                ](),
                batch_size,
                vocab_size,
                ignore_index,
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()

            var L = logits.to_tile_tensor[DType.int32]()
            var T = targets.to_tile_tensor[DType.int32]()
            var O = output.to_tile_tensor[DType.int32]()

            # Pre-compute the count of valid (non-ignored) samples so the GPU
            # kernel can directly produce the mean loss in a single pass. The
            # targets tensor metadata is accessible on host; only the
            # ignore_index scalar is needed for the comparison.
            var count = 0
            for i in range(batch_size):
                if Int(T.ptr[i]) != ignore_index:
                    count += 1

            if count == 0:
                return

            var inv_count = Scalar[DType.float32](1) / Scalar[DType.float32](
                count
            )

            comptime BLOCK_SIZE = 256

            var compiled_func = gpu_ctx.compile_function[
                _cross_entropy_fwd_gpu[
                    BLOCK_SIZE,
                    L.dtype,
                    T.dtype,
                    O.dtype,
                    L.LayoutType,
                    T.LayoutType,
                    O.LayoutType,
                ],
                _cross_entropy_fwd_gpu[
                    BLOCK_SIZE,
                    L.dtype,
                    T.dtype,
                    O.dtype,
                    L.LayoutType,
                    T.LayoutType,
                    O.LayoutType,
                ],
            ]()

            gpu_ctx.enqueue_function(
                compiled_func,
                L,
                T,
                O,
                batch_size,
                vocab_size,
                ignore_index,
                inv_count,
                grid_dim=min(batch_size, 1024),
                block_dim=BLOCK_SIZE,
            )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        logits: InputTensor[dtype=dtype, rank=2, ...],
        targets: InputTensor[dtype=DType.int64, rank=1, ...],
        ignore_index_tensor: InputTensor[dtype=DType.int64, rank=1, ...],
    ) -> IndexList[1]:
        return IndexList[1](1)


# ===----------------------------------------------------------------------=== #
# Backward kernel: softmax - one_hot
# ===----------------------------------------------------------------------=== #


def _cross_entropy_bwd_gpu[
    BLOCK_SIZE: Int,
    logits_dtype: DType,
    targets_dtype: DType,
    grad_dtype: DType,
    LogitsLayoutType: layout.TensorLayout,
    TargetsLayoutType: layout.TensorLayout,
    GradLayoutType: layout.TensorLayout,
    logits_origin: ImmutOrigin,
    targets_origin: ImmutOrigin,
    grad_origin: MutOrigin,
    accum_type: DType = get_accum_type[logits_dtype](),
](
    logits: layout.TileTensor[
        logits_dtype, LogitsLayoutType, logits_origin
    ],
    targets: layout.TileTensor[
        targets_dtype, TargetsLayoutType, targets_origin
    ],
    grad_logits: layout.TileTensor[
        grad_dtype, GradLayoutType, grad_origin
    ],
    batch_size: Int,
    vocab_size: Int,
):
    """GPU kernel for cross-entropy backward pass.

    Each block processes one row. Computes softmax(logits) - one_hot(targets),
    scaled by 1/batch_size for mean reduction consistency.
    """
    var tid = Int(thread_idx.x)
    var inv_batch = Scalar[accum_type](1) / Scalar[accum_type](batch_size)

    for row_idx in range(Int(block_idx.x), batch_size, Int(grid_dim.x)):
        var target = Int(targets.ptr[row_idx])
        var row_base = row_idx * vocab_size

        # Pass 1: row max.
        var row_max = Scalar[accum_type].MIN
        for col in range(tid, vocab_size, BLOCK_SIZE):
            var val = logits.ptr[row_base + col].cast[accum_type]()
            row_max = max(row_max, val)

        row_max = block.max[block_size=BLOCK_SIZE](row_max)

        # Pass 2: sum(exp(logit - max)).
        var exp_sum = Scalar[accum_type](0)
        for col in range(tid, vocab_size, BLOCK_SIZE):
            var val = logits.ptr[row_base + col].cast[accum_type]()
            exp_sum += exp(val - row_max)

        exp_sum = block.sum[block_size=BLOCK_SIZE](exp_sum)
        var inv_sum = Scalar[accum_type](1) / exp_sum

        # Pass 3: grad = (softmax - one_hot) / batch_size.
        for col in range(tid, vocab_size, BLOCK_SIZE):
            var val = logits.ptr[row_base + col].cast[accum_type]()
            var softmax_val = exp(val - row_max) * inv_sum
            var indicator = Scalar[accum_type](1) if col == target else Scalar[
                accum_type
            ](0)
            grad_logits.ptr[row_base + col] = (
                (softmax_val - indicator) * inv_batch
            ).cast[grad_dtype]()


def _cross_entropy_bwd_cpu[
    logits_dtype: DType,
    targets_dtype: DType,
    grad_dtype: DType,
](
    logits_ptr: UnsafePointer[Scalar[logits_dtype]],
    targets_ptr: UnsafePointer[Scalar[targets_dtype]],
    grad_ptr: UnsafePointer[Scalar[grad_dtype]],
    batch_size: Int,
    vocab_size: Int,
):
    """CPU fallback for cross-entropy backward pass."""
    comptime accum_type = get_accum_type[logits_dtype]()
    var inv_batch = Scalar[accum_type](1) / Scalar[accum_type](batch_size)

    for row in range(batch_size):
        var target = Int(targets_ptr[row])
        var row_base = row * vocab_size

        var row_max = Scalar[accum_type].MIN
        for col in range(vocab_size):
            var val = logits_ptr[row_base + col].cast[accum_type]()
            row_max = max(row_max, val)

        var exp_sum = Scalar[accum_type](0)
        for col in range(vocab_size):
            var val = logits_ptr[row_base + col].cast[accum_type]()
            exp_sum += exp(val - row_max)

        var inv_sum = Scalar[accum_type](1) / exp_sum

        for col in range(vocab_size):
            var val = logits_ptr[row_base + col].cast[accum_type]()
            var softmax_val = exp(val - row_max) * inv_sum
            var indicator = Scalar[accum_type](1) if col == target else Scalar[
                accum_type
            ](0)
            grad_ptr[row_base + col] = (
                (softmax_val - indicator) * inv_batch
            ).cast[grad_dtype]()


@compiler.register("training.cross_entropy_bwd")
struct CrossEntropyBwd:
    """Fused softmax - one_hot backward pass for cross-entropy loss.

    Computes d_logits = (softmax(logits) - one_hot(targets)) / batch_size
    in a single fused pass, avoiding materialization of intermediate softmax.

    Tensor Shapes:
        - logits: (batch, vocab) - Unnormalized log-probabilities (same as fwd).
        - targets: (batch,) - Ground-truth class indices.
        - grad_logits: (batch, vocab) - Output gradient w.r.t. logits.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        grad_logits: OutputTensor[dtype=dtype, rank=2, ...],
        logits: InputTensor[dtype=dtype, rank=2, ...],
        targets: InputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var batch_size = logits.dim_size(0)
        var vocab_size = logits.dim_size(1)

        comptime if is_cpu[target]():
            _cross_entropy_bwd_cpu[dtype, DType.int64, dtype](
                logits.to_tile_tensor[DType.int32]().ptr.bitcast[
                    Scalar[dtype]
                ](),
                targets.to_tile_tensor[DType.int32]().ptr.bitcast[
                    Scalar[DType.int64]
                ](),
                grad_logits.to_tile_tensor[DType.int32]().ptr.bitcast[
                    Scalar[dtype]
                ](),
                batch_size,
                vocab_size,
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()

            var L = logits.to_tile_tensor[DType.int32]()
            var T = targets.to_tile_tensor[DType.int32]()
            var G = grad_logits.to_tile_tensor[DType.int32]()

            comptime BLOCK_SIZE = 256

            var compiled_func = gpu_ctx.compile_function[
                _cross_entropy_bwd_gpu[
                    BLOCK_SIZE,
                    L.dtype,
                    T.dtype,
                    G.dtype,
                    L.LayoutType,
                    T.LayoutType,
                    G.LayoutType,
                ],
                _cross_entropy_bwd_gpu[
                    BLOCK_SIZE,
                    L.dtype,
                    T.dtype,
                    G.dtype,
                    L.LayoutType,
                    T.LayoutType,
                    G.LayoutType,
                ],
            ]()

            gpu_ctx.enqueue_function(
                compiled_func,
                L,
                T,
                G,
                batch_size,
                vocab_size,
                grid_dim=min(batch_size, 1024),
                block_dim=BLOCK_SIZE,
            )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        logits: InputTensor[dtype=dtype, rank=2, ...],
        targets: InputTensor[dtype=DType.int64, rank=1, ...],
    ) -> IndexList[2]:
        return logits.shape()
