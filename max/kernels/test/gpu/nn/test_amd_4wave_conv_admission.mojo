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
"""AMD convolution dispatch admission and exact-output boundary checks."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.conv.conv import conv_gpu
from nn.conv.gpu.amd.dispatch import dispatch_amd_4wave_conv2d

comptime D = DType.float16
comptime GUARD = 128


def run_row[
    OH: Int,
    OW: Int,
    C: Int,
    R: Int,
    stride: Int,
    pad: Int,
    dynamic: Bool,
    expected: Bool,
    bm: Int = 0,
    bn: Int = 0,
    bk: Int = 0,
](ctx: DeviceContext, label: StaticString) raises:
    comptime H = (OH - 1) * stride + R - 2 * pad
    comptime W = (OW - 1) * stride + R - 2 * pad
    comptime NI = H * W * C
    comptime NF = C * C * R * R
    comptime NO = OH * OW * C
    comptime assert H > 0 and W > 0
    var ib = ctx.enqueue_create_buffer[D](NI + 2 * GUARD)
    var fb = ctx.enqueue_create_buffer[D](NF + 2 * GUARD)
    var ob = ctx.enqueue_create_buffer[D](NO + 2 * GUARD)
    var ih = ctx.enqueue_create_host_buffer[D](NI + 2 * GUARD)
    var fh = ctx.enqueue_create_host_buffer[D](NF + 2 * GUARD)
    var oh = ctx.enqueue_create_host_buffer[D](NO + 2 * GUARD)
    for i in range(NI + 2 * GUARD):
        ih[i] = 123 if i < GUARD or i >= GUARD + NI else 1
    for i in range(NF + 2 * GUARD):
        fh[i] = 123 if i < GUARD or i >= GUARD + NF else 1
    for i in range(NO + 2 * GUARD):
        oh[i] = 123 if i < GUARD or i >= GUARD + NO else bitcast[D](
            UInt16(0x7E00)
        )
    ctx.enqueue_copy(ib, ih)
    ctx.enqueue_copy(fb, fh)
    ctx.enqueue_copy(ob, oh)
    var f = TileTensor(
        fb.unsafe_ptr().unsafe_offset(GUARD),
        row_major(Idx[C], Idx[C], Idx[R], Idx[R]),
    )
    var strides = (stride, stride)
    var dilation = (1, 1)
    var pads = (pad, pad)

    def run_views(
        x: TileTensor[mut=True, D, address_space=.GENERIC, ...],
        y: TileTensor[mut=True, D, address_space=.GENERIC, ...],
    ) raises {imm}:
        comptime if dynamic:
            comptime assert (
                x.static_shape[0] < 0
                and x.static_shape[1] < 0
                and x.static_shape[2] < 0
            )
        else:
            comptime assert (
                x.static_shape[0] == 1
                and x.static_shape[1] == H
                and x.static_shape[2] == W
            )
        var hit = dispatch_amd_4wave_conv2d[
            input_type=D,
            filter_type=D,
            output_type=D,
            filter_is_fcrs=True,
            block_m_override=bm,
            block_n_override=bn,
            block_k_override=bk,
        ](x, f, y, strides, dilation, pads, 1, ctx)
        if hit != expected:
            raise Error("unexpected direct dispatch decision")
        comptime if not expected:
            conv_gpu[filter_is_fcrs=True](
                x, f, y, strides, dilation, (pad, pad, pad, pad), 1, ctx
            )

    comptime if dynamic:
        # Force runtime dimensions even though this probe has known concrete sizes.
        var n = 1
        var h = H
        var w = W
        var out_h = OH
        var out_w = OW
        var x = TileTensor(
            ib.unsafe_ptr().unsafe_offset(GUARD), row_major(n, h, w, Idx[C])
        )
        var y = TileTensor(
            ob.unsafe_ptr().unsafe_offset(GUARD),
            row_major(n, out_h, out_w, Idx[C]),
        )
        run_views(x, y)
    else:
        var x = TileTensor(
            ib.unsafe_ptr().unsafe_offset(GUARD),
            row_major(Idx[1], Idx[H], Idx[W], Idx[C]),
        )
        var y = TileTensor(
            ob.unsafe_ptr().unsafe_offset(GUARD),
            row_major(Idx[1], Idx[OH], Idx[OW], Idx[C]),
        )
        run_views(x, y)
    ctx.enqueue_copy(oh, ob)
    ctx.enqueue_copy(ih, ib)
    ctx.enqueue_copy(fh, fb)
    ctx.synchronize()
    for y in range(OH):
        for x in range(OW):
            var count = 0
            for r in range(R):
                for s in range(R):
                    var iy = y * stride - pad + r
                    var ix = x * stride - pad + s
                    if iy >= 0 and iy < H and ix >= 0 and ix < W:
                        count += 1
            var reference = Float32(count * C)
            for channel in range(C):
                var actual = Float32(oh[GUARD + (y * OW + x) * C + channel])
                if actual != reference:
                    print(
                        t"MISMATCH row={label} y={y} x={x} channel={channel}"
                        t" actual={actual} reference={reference}"
                    )
                    raise Error("whole-output exact reference mismatch")
    for i in range(GUARD):
        if (
            Float32(ih[i]) != 123
            or Float32(ih[GUARD + NI + i]) != 123
            or Float32(fh[i]) != 123
            or Float32(fh[GUARD + NF + i]) != 123
            or Float32(oh[i]) != 123
            or Float32(oh[GUARD + NO + i]) != 123
        ):
            raise Error("input, filter, or output guard corrupted")
    print(
        t"MATRIX_PASS row={label} hit={expected} dynamic={dynamic}"
        t" stride={stride} pad={pad} output_elements={NO} guards={GUARD}"
    )


def pair[stride: Int, pad: Int](ctx: DeviceContext) raises:
    # BM64 gives rows_per_iteration=32: 32//8+1 < 8 and 32//5+1 < 13.
    run_row[8, 8, 64, 1, stride, pad, False, True, 64, 64](ctx, "static_m64")
    run_row[13, 5, 64, 1, stride, pad, False, False, 64, 64](ctx, "static_m65")


def main() raises:
    with DeviceContext() as ctx:
        pair[1, 0](ctx)
        pair[1, 1](ctx)
        pair[1, 2](ctx)
        pair[2, 0](ctx)
        pair[2, 1](ctx)
        pair[2, 2](ctx)
        run_row[9, 7, 64, 1, 1, 0, False, False, 64, 64](ctx, "static_m63")
        run_row[9, 7, 64, 1, 1, 0, True, True, 64, 64](ctx, "dynamic_m63")
        run_row[8, 8, 64, 1, 1, 0, True, True, 64, 64](ctx, "dynamic_m64")
        run_row[13, 5, 64, 1, 1, 0, True, True, 64, 64](ctx, "dynamic_m65")
        # 9x9 is ragged, within the 4096 bound; 32//9+1 < 9.
        run_row[9, 9, 384, 3, 1, 1, False, True](ctx, "c384_default")
        run_row[9, 9, 384, 3, 1, 1, False, False, 64, 64](
            ctx, "c384_bm_bn_override"
        )
        run_row[9, 9, 384, 3, 1, 1, False, True, 0, 0, 64](
            ctx, "c384_bk_override"
        )
