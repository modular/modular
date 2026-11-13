# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Launches each kernel a GPT-2 forward pass is built from (the bias
projection, layer norm, GELU, softmax and causal attention) through the typed
`Queue.execute`, and checks it against a host reference.

The shapes are small and the kernels naive: what the suite covers is that each
kernel's argument shapes, several `TileTensor`s with static and run-time dims,
inline scalars between them and pointer outputs, reach the kernel intact on
every HAL device.
"""

from std.math import exp, sqrt, tanh

from layout import Idx, TensorLayout, TileTensor, row_major
from max._driver import Driver, Context
from max._driver.max_license.hal._machine import DeviceSpec
from max._driver.max_license.hal.queue import Queue

from ukl_test_support import (
    alloc_zeroed,
    device_ptr,
    driver_plugins,
    expect_close,
    first_item,
    item_stride,
    launch,
    upload,
)

comptime F32 = DType.float32
comptime ROWS = 8
comptime HIDDEN = 16
comptime PROJ = 24
comptime HEAD_DIM = 8
comptime EPSILON = Float32(1e-5)
comptime SQRT_2_OVER_PI = 0.7978845608028654
# Devices may approximate `exp`, `tanh` and `sqrt` differently from the host.
comptime ATOL = 1e-5
comptime RTOL = 1e-4
# NVIDIA's float32 `tanh` is `tanh.approx.f32`, accurate only to about 2^-11
# relative, and `1 + tanh` cancels for negative inputs, so an error near 1e-3
# survives into GELU's output.
comptime GELU_ATOL = 2e-3


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


def matmul_bias[
    InL: TensorLayout, WL: TensorLayout, BL: TensorLayout, OutL: TensorLayout
](
    inp: TileTensor[F32, InL, ImmUntrackedOrigin],
    weight: TileTensor[F32, WL, ImmUntrackedOrigin],
    bias: TileTensor[F32, BL, ImmUntrackedOrigin],
    output: TileTensor[F32, OutL, MutUntrackedOrigin],
):
    """Writes `inp @ weight + bias`, as GPT-2's QKV, attention output and MLP
    projections do."""
    var cols = Int(output.dim[1]())
    var depth = Int(inp.dim[1]())
    for i in range(first_item(), Int(output.dim[0]()) * cols, item_stride()):
        var row = i // cols
        var col = i % cols
        var acc = rebind[Float32](bias[col])
        for k in range(depth):
            acc += rebind[Float32](inp[row, k]) * rebind[Float32](
                weight[k, col]
            )
        output[row, col] = rebind[output.ElementType](acc)


def layer_norm[
    InL: TensorLayout, ParamL: TensorLayout
](
    src: TileTensor[F32, InL, ImmUntrackedOrigin],
    gamma: TileTensor[F32, ParamL, ImmUntrackedOrigin],
    beta: TileTensor[F32, ParamL, ImmUntrackedOrigin],
    epsilon: Float32,
    output: MutPointer[Float32, MutUntrackedOrigin],
    row_length: Int32,
):
    """Normalizes each row of `src` and scales and shifts it by `gamma` and
    `beta`, writing rows of `row_length` to `output`."""
    var length = Int(row_length)
    for row in range(first_item(), Int(src.dim[0]()), item_stride()):
        var mean = Float32(0)
        for c in range(length):
            mean += rebind[Float32](src[row, c])
        mean /= Float32(length)
        var variance = Float32(0)
        for c in range(length):
            var centered = rebind[Float32](src[row, c]) - mean
            variance += centered * centered
        var inv_std = Float32(1) / sqrt(variance / Float32(length) + epsilon)
        for c in range(length):
            var normed = (rebind[Float32](src[row, c]) - mean) * inv_std
            output[unsafe_offset=row * length + c] = normed * rebind[Float32](
                gamma[c]
            ) + rebind[Float32](beta[c])


def gelu[
    n: Int
](
    x: ImmPointer[Float32, ImmUntrackedOrigin],
    output: MutPointer[Float32, MutUntrackedOrigin],
):
    """Applies GPT-2's tanh approximation of GELU to each of `x`'s `n`
    values."""
    for i in range(first_item(), n, item_stride()):
        var v = x[unsafe_offset=i]
        output[unsafe_offset=i] = (
            0.5 * v * (1 + tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))
        )


def softmax[
    InL: TensorLayout, OutL: TensorLayout
](
    x: TileTensor[F32, InL, ImmUntrackedOrigin],
    output: TileTensor[F32, OutL, MutUntrackedOrigin],
):
    """Writes the softmax of each row of `x`."""
    var cols = Int(x.dim[1]())
    for row in range(first_item(), Int(x.dim[0]()), item_stride()):
        var row_max = rebind[Float32](x[row, 0])
        for c in range(1, cols):
            var v = rebind[Float32](x[row, c])
            if v > row_max:
                row_max = v
        var denom = Float32(0)
        for c in range(cols):
            denom += exp(rebind[Float32](x[row, c]) - row_max)
        for c in range(cols):
            output[row, c] = rebind[output.ElementType](
                exp(rebind[Float32](x[row, c]) - row_max) / denom
            )


@inline(.always)
def _dot_rows[
    AL: TensorLayout, BL: TensorLayout
](
    a: TileTensor[F32, AL, ImmUntrackedOrigin],
    a_row: Int,
    b: TileTensor[F32, BL, ImmUntrackedOrigin],
    b_row: Int,
) -> Float32:
    var acc = Float32(0)
    for d in range(Int(a.dim[1]())):
        acc += rebind[Float32](a[a_row, d]) * rebind[Float32](b[b_row, d])
    return acc


def attention[
    QL: TensorLayout, KL: TensorLayout, VL: TensorLayout, OL: TensorLayout
](
    q: TileTensor[F32, QL, ImmUntrackedOrigin],
    k: TileTensor[F32, KL, ImmUntrackedOrigin],
    v: TileTensor[F32, VL, ImmUntrackedOrigin],
    scale: Float32,
    output: TileTensor[F32, OL, MutUntrackedOrigin],
):
    """Computes single-head causal attention, where each query attends to
    itself and the tokens before it.

    Each output value recomputes its scores rather than accumulating into
    `output`, which a device running every item on each thread can't do.
    """
    var head_dim = Int(q.dim[1]())
    for t in range(first_item(), Int(q.dim[0]()), item_stride()):
        var row_max = _dot_rows(q, t, k, 0) * scale
        for j in range(1, t + 1):
            var score = _dot_rows(q, t, k, j) * scale
            if score > row_max:
                row_max = score
        var denom = Float32(0)
        for j in range(t + 1):
            denom += exp(_dot_rows(q, t, k, j) * scale - row_max)
        for d in range(head_dim):
            var acc = Float32(0)
            for j in range(t + 1):
                var weight = exp(_dot_rows(q, t, k, j) * scale - row_max)
                acc += weight * rebind[Float32](v[j, d])
            output[t, d] = rebind[output.ElementType](acc / denom)


# ===-----------------------------------------------------------------------===#
# Host references
# ===-----------------------------------------------------------------------===#


def _values(length: Int, mul: Int, shift: Int, scale: Float32) -> List[Float32]:
    """Returns a deterministic spread of `length` values in
    `[-8 * scale, 8 * scale]`."""
    var values = List[Float32](capacity=length)
    for i in range(length):
        values.append(Float32((i * mul + shift) % 17 - 8) * scale)
    return values^


def _matmul_bias_expected(
    inp: List[Float32], weight: List[Float32], bias: List[Float32]
) -> List[Float32]:
    var expected = List[Float32](capacity=ROWS * PROJ)
    for row in range(ROWS):
        for col in range(PROJ):
            var acc = Float64(bias[col])
            for k in range(HIDDEN):
                acc += Float64(inp[row * HIDDEN + k]) * Float64(
                    weight[k * PROJ + col]
                )
            expected.append(Float32(acc))
    return expected^


def _layer_norm_expected(
    src: List[Float32], gamma: List[Float32], beta: List[Float32]
) -> List[Float32]:
    var expected = List[Float32](capacity=ROWS * HIDDEN)
    for row in range(ROWS):
        var mean = Float64(0)
        for c in range(HIDDEN):
            mean += Float64(src[row * HIDDEN + c])
        mean /= Float64(HIDDEN)
        var variance = Float64(0)
        for c in range(HIDDEN):
            var centered = Float64(src[row * HIDDEN + c]) - mean
            variance += centered * centered
        var inv_std = 1 / sqrt(variance / Float64(HIDDEN) + Float64(EPSILON))
        for c in range(HIDDEN):
            var normed = (Float64(src[row * HIDDEN + c]) - mean) * inv_std
            expected.append(
                Float32(normed * Float64(gamma[c]) + Float64(beta[c]))
            )
    return expected^


def _gelu_expected(x: List[Float32]) -> List[Float32]:
    var expected = List[Float32](capacity=len(x))
    for i in range(len(x)):
        var v = Float64(x[i])
        expected.append(
            Float32(
                0.5
                * v
                * (1 + tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))
            )
        )
    return expected^


def _softmax(row: List[Float64]) -> List[Float64]:
    var row_max = row[0]
    for v in row:
        if v > row_max:
            row_max = v
    var denom = Float64(0)
    for v in row:
        denom += exp(v - row_max)
    var probs = List[Float64](capacity=len(row))
    for v in row:
        probs.append(exp(v - row_max) / denom)
    return probs^


def _softmax_expected(x: List[Float32], cols: Int) -> List[Float32]:
    var expected = List[Float32](capacity=len(x))
    for row in range(len(x) // cols):
        var values = List[Float64](capacity=cols)
        for c in range(cols):
            values.append(Float64(x[row * cols + c]))
        for p in _softmax(values):
            expected.append(Float32(p))
    return expected^


def _attention_expected(
    q: List[Float32], k: List[Float32], v: List[Float32], scale: Float32
) -> List[Float32]:
    var expected = List[Float32](capacity=ROWS * HEAD_DIM)
    for t in range(ROWS):
        var scores = List[Float64](capacity=t + 1)
        for j in range(t + 1):
            var score = Float64(0)
            for d in range(HEAD_DIM):
                score += Float64(q[t * HEAD_DIM + d]) * Float64(
                    k[j * HEAD_DIM + d]
                )
            scores.append(score * Float64(scale))
        var probs = _softmax(scores)
        for d in range(HEAD_DIM):
            var acc = Float64(0)
            for j in range(t + 1):
                acc += probs[j] * Float64(v[j * HEAD_DIM + d])
            expected.append(Float32(acc))
    return expected^


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def test_matmul_bias[
    spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes four `TileTensor`s, the input and output with a row count known
    only at run time, so their dims travel with them."""
    print("matmul_bias")
    var inp_host = _values(ROWS * HIDDEN, 7, 3, 0.125)
    var weight_host = _values(HIDDEN * PROJ, 5, 1, 0.0625)
    var bias_host = _values(PROJ, 3, 2, 0.25)
    var inp = upload(ctx, queue, inp_host)
    var weight = upload(ctx, queue, weight_host)
    var bias = upload(ctx, queue, bias_host)
    var out = alloc_zeroed[F32](ctx, queue, ROWS * PROJ)

    var rows = ROWS
    var inp_tile = TileTensor(
        ptr=device_ptr[F32](ctx, inp), layout=row_major(rows, Idx[HIDDEN])
    )
    var weight_tile = TileTensor(
        ptr=device_ptr[F32](ctx, weight), layout=row_major[HIDDEN, PROJ]()
    )
    var bias_tile = TileTensor(
        ptr=device_ptr[F32](ctx, bias), layout=row_major[PROJ]()
    )
    var out_tile = TileTensor(
        ptr=device_ptr[F32, mut=True](ctx, out),
        layout=row_major(rows, Idx[PROJ]),
    )
    launch[
        matmul_bias[
            inp_tile.LayoutType,
            weight_tile.LayoutType,
            bias_tile.LayoutType,
            out_tile.LayoutType,
        ]
    ](queue, ROWS * PROJ, inp_tile, weight_tile, bias_tile, out_tile)

    expect_close(
        queue,
        out,
        _matmul_bias_expected(inp_host, weight_host, bias_host),
        "matmul_bias",
        atol=ATOL,
        rtol=RTOL,
    )
    ctx.free_sync(out^)
    ctx.free_sync(bias^)
    ctx.free_sync(weight^)
    ctx.free_sync(inp^)


def test_layer_norm[
    spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes two inline scalars, a `Float32` epsilon and an `Int32` row
    length, among `TileTensor`s and a `Buffer` for the output pointer."""
    print("layer_norm")
    var src_host = _values(ROWS * HIDDEN, 11, 4, 0.25)
    var gamma_host = _values(HIDDEN, 3, 12, 0.125)
    var beta_host = _values(HIDDEN, 5, 6, 0.0625)
    var src = upload(ctx, queue, src_host)
    var gamma = upload(ctx, queue, gamma_host)
    var beta = upload(ctx, queue, beta_host)
    var out = alloc_zeroed[F32](ctx, queue, ROWS * HIDDEN)

    var src_tile = TileTensor(
        ptr=device_ptr[F32](ctx, src), layout=row_major[ROWS, HIDDEN]()
    )
    var gamma_tile = TileTensor(
        ptr=device_ptr[F32](ctx, gamma), layout=row_major[HIDDEN]()
    )
    var beta_tile = TileTensor(
        ptr=device_ptr[F32](ctx, beta), layout=row_major[HIDDEN]()
    )
    launch[layer_norm[src_tile.LayoutType, gamma_tile.LayoutType]](
        queue,
        ROWS,
        src_tile,
        gamma_tile,
        beta_tile,
        EPSILON,
        out,
        Int32(HIDDEN),
    )

    expect_close(
        queue,
        out,
        _layer_norm_expected(src_host, gamma_host, beta_host),
        "layer_norm",
        atol=ATOL,
        rtol=RTOL,
    )
    ctx.free_sync(out^)
    ctx.free_sync(beta^)
    ctx.free_sync(gamma^)
    ctx.free_sync(src^)


def test_gelu[spec: DeviceSpec](ctx: Context[spec], queue: Queue[spec]) raises:
    """Fills the pointers with a `Buffer` and a `BufferView`."""
    print("gelu")
    comptime n = ROWS * PROJ
    var x_host = _values(n, 13, 5, 0.5)
    var x = upload(ctx, queue, x_host)
    var out = alloc_zeroed[F32](ctx, queue, n)

    launch[gelu[n]](queue, n, x, out.view())

    expect_close(
        queue, out, _gelu_expected(x_host), "gelu", atol=GELU_ATOL, rtol=RTOL
    )
    ctx.free_sync(out^)
    ctx.free_sync(x^)


def test_softmax[
    spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes `TileTensor`s whose every dim is known only at run time."""
    print("softmax")
    var x_host = _values(ROWS * PROJ, 7, 9, 0.25)
    var x = upload(ctx, queue, x_host)
    var out = alloc_zeroed[F32](ctx, queue, ROWS * PROJ)

    var rows = ROWS
    var cols = PROJ
    var x_tile = TileTensor(
        ptr=device_ptr[F32](ctx, x), layout=row_major(rows, cols)
    )
    var out_tile = TileTensor(
        ptr=device_ptr[F32, mut=True](ctx, out), layout=row_major(rows, cols)
    )
    launch[softmax[x_tile.LayoutType, out_tile.LayoutType]](
        queue, ROWS, x_tile, out_tile
    )

    expect_close(
        queue,
        out,
        _softmax_expected(x_host, PROJ),
        "softmax",
        atol=ATOL,
        rtol=RTOL,
    )
    ctx.free_sync(out^)
    ctx.free_sync(x^)


def test_attention[
    spec: DeviceSpec
](ctx: Context[spec], queue: Queue[spec]) raises:
    """Passes four `TileTensor`s with an inline `Float32` scale between
    them."""
    print("attention")
    var q_host = _values(ROWS * HEAD_DIM, 5, 2, 0.25)
    var k_host = _values(ROWS * HEAD_DIM, 7, 8, 0.25)
    var v_host = _values(ROWS * HEAD_DIM, 3, 1, 0.125)
    var q = upload(ctx, queue, q_host)
    var k = upload(ctx, queue, k_host)
    var v = upload(ctx, queue, v_host)
    var out = alloc_zeroed[F32](ctx, queue, ROWS * HEAD_DIM)
    var scale = Float32(1) / sqrt(Float32(HEAD_DIM))

    var q_tile = TileTensor(
        ptr=device_ptr[F32](ctx, q), layout=row_major[ROWS, HEAD_DIM]()
    )
    var k_tile = TileTensor(
        ptr=device_ptr[F32](ctx, k), layout=row_major[ROWS, HEAD_DIM]()
    )
    var v_tile = TileTensor(
        ptr=device_ptr[F32](ctx, v), layout=row_major[ROWS, HEAD_DIM]()
    )
    var out_tile = TileTensor(
        ptr=device_ptr[F32, mut=True](ctx, out),
        layout=row_major[ROWS, HEAD_DIM](),
    )
    launch[
        attention[
            q_tile.LayoutType,
            k_tile.LayoutType,
            v_tile.LayoutType,
            out_tile.LayoutType,
        ]
    ](queue, ROWS, q_tile, k_tile, v_tile, scale, out_tile)

    expect_close(
        queue,
        out,
        _attention_expected(q_host, k_host, v_host, scale),
        "attention",
        atol=ATOL,
        rtol=RTOL,
    )
    ctx.free_sync(out^)
    ctx.free_sync(v^)
    ctx.free_sync(k^)
    ctx.free_sync(q^)


def main() raises:
    var driver = Driver.create(driver_plugins())
    var device = driver[].get_device[0]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()

    test_matmul_bias(ctx[], queue[])
    test_layer_norm(ctx[], queue[])
    test_gelu(ctx[], queue[])
    test_softmax(ctx[], queue[])
    test_attention(ctx[], queue[])
