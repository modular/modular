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

from max.gpu.host import DeviceContext
from layout import DefaultEngine, Idx, TileTensor, row_major
from nn.roi_align import roi_align_nhwc
from nn.roi_align_gpu import roi_align_gpu
from std.testing import assert_almost_equal


@inline(.never)
def test_roi_align_gpu[
    dtype: DType,
    aligned: Bool,
    mode: StaticString,
    n: Int,
    h: Int,
    w: Int,
    c: Int,
    m: Int,
    pooled_h: Int,
    pooled_w: Int,
](
    spatial_scale: Float32,
    sampling_ratio: Float32,
    ctx: DeviceContext,
    verbose: Bool = False,
) raises:
    print("== test_roi_align_gpu aligned=" + String(aligned) + " mode=" + mode)

    comptime in_layout = row_major[n, h, w, c]()
    comptime out_layout = row_major[m, pooled_h, pooled_w, c]()
    comptime roi_layout = row_major[m, 5]()

    # Host input: sequential values.
    var input_host = ctx.enqueue_create_host_buffer[dtype](in_layout.product())
    ctx.synchronize()
    var input = TileTensor(input_host, in_layout)
    var input_flat = input.reshape(row_major(in_layout.product()))
    for i in range(in_layout.product()):
        input_flat[input.layout(i)] = Scalar[dtype](i)

    # Host rois: a spread of boxes. The first ROI starts at -1 (x0 = y0 = -1
    # for ri = 0), which exercises the negative-start clamp (y = max(y, 0));
    # for large m the ROI ends (x1 = y1 = ri + 3) run far past the input,
    # which exercises the out-of-range early-out (y > height / x > width).
    var rois_host = ctx.enqueue_create_host_buffer[dtype](roi_layout.product())
    ctx.synchronize()
    var rois = TileTensor(rois_host, roi_layout)
    for ri in range(m):
        rois[ri, 0] = Scalar[dtype](ri % n)
        rois[ri, 1] = Scalar[dtype](Float32(ri) - 1.0)  # x0
        rois[ri, 2] = Scalar[dtype](Float32(ri) - 1.0)  # y0
        rois[ri, 3] = Scalar[dtype](Float32(ri) + 3.0)  # x1
        rois[ri, 4] = Scalar[dtype](Float32(ri) + 3.0)  # y1

    # Host output, zeroed.
    var output_host = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )
    ctx.synchronize()
    var output = TileTensor(output_host, out_layout).fill(0)

    # Device buffers.
    var in_device = ctx.enqueue_create_buffer[dtype](in_layout.product())
    var rois_device = ctx.enqueue_create_buffer[dtype](roi_layout.product())
    var out_device = ctx.enqueue_create_buffer[dtype](out_layout.product())
    ctx.enqueue_copy(in_device, input_host)
    ctx.enqueue_copy(rois_device, rois_host)
    ctx.enqueue_copy(out_device, output_host)

    var in_dev_tt = TileTensor(in_device, in_layout)
    var rois_dev_tt = TileTensor(rois_device, roi_layout)
    var out_dev_tt = TileTensor(out_device, out_layout)

    roi_align_gpu[dtype, aligned, mode](
        out_dev_tt,
        in_dev_tt,
        rois_dev_tt,
        pooled_h,
        pooled_w,
        spatial_scale,
        sampling_ratio,
        ctx,
    )

    ctx.enqueue_copy(output_host, out_device)
    ctx.synchronize()

    # CPU reference into a fresh host buffer.
    var output_cpu_host = ctx.enqueue_create_host_buffer[dtype](
        out_layout.product()
    )
    ctx.synchronize()
    var output_cpu = TileTensor(output_cpu_host, out_layout).fill(0)
    roi_align_nhwc[aligned=aligned, mode=mode](
        output_cpu,
        input,
        rois,
        pooled_h,
        pooled_w,
        spatial_scale,
        sampling_ratio,
    )

    if verbose:
        print(output)
        print(output_cpu)

    for i in range(out_layout.product()):
        assert_almost_equal(
            output[i],
            output_cpu[i],
            atol=1e-4,
            rtol=1e-4,
            msg=String("roi_align mismatch at flat index ", i),
        )

    print(
        "PASS: aligned="
        + String(aligned)
        + " mode="
        + mode
        + " n="
        + String(n)
        + " c="
        + String(c)
    )

    _ = in_device
    _ = rois_device
    _ = out_device


def main() raises:
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        # 10x10 input, 1 channel, 4 ROIs, 5x5 pooled output.
        test_roi_align_gpu[dtype, False, "AVG", 2, 10, 10, 1, 4, 5, 5](
            1.0, 2.0, ctx
        )
        test_roi_align_gpu[dtype, True, "AVG", 2, 10, 10, 1, 4, 5, 5](
            1.0, 2.0, ctx
        )
        test_roi_align_gpu[dtype, False, "MAX", 2, 10, 10, 1, 4, 5, 5](
            1.0, 2.0, ctx
        )
        test_roi_align_gpu[dtype, True, "MAX", 2, 10, 10, 1, 4, 5, 5](
            1.0, 2.0, ctx
        )
        # Multi-channel (4) with a smaller sampling ratio and a fractional
        # spatial scale.
        test_roi_align_gpu[dtype, False, "AVG", 2, 8, 8, 4, 3, 4, 4](
            0.5, 1.0, ctx
        )
        test_roi_align_gpu[dtype, True, "MAX", 2, 8, 8, 4, 3, 4, 4](
            0.5, 1.0, ctx
        )
        # Many ROIs so total_rows exceeds sm_count * block_dim.y: forces
        # rows_per_block > block_dim.y and exercises the row grid-stride.
        test_roi_align_gpu[dtype, False, "AVG", 2, 10, 10, 1, 200, 5, 5](
            1.0, 2.0, ctx
        )
        # More than 32 channels: exercises the channel grid-stride.
        test_roi_align_gpu[dtype, True, "MAX", 2, 10, 10, 40, 4, 5, 5](
            1.0, 2.0, ctx
        )
