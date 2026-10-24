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
"""GPU tests for the NMS kernel (the `mo.non_maximum_suppression` op).

Each case builds boxes and scores on the host and routes them through
the target dispatch, comparing both the CPU result and the GPU result
against a serial reference that follows the shared total order (score
descending, index ascending) pinned by both kernels and the shape
function. Outputs are allocated at the exact row count with a poisoned
guard band past it, so a missing row or an out-of-bounds write fails
loudly.
"""

comptime _GUARD = 8

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.nms import non_max_suppression

from std.math import nan
from std.testing import assert_equal


struct BoxCoords[dtype: DType](TrivialRegisterPassable):
    var y1: Scalar[Self.dtype]
    var x1: Scalar[Self.dtype]
    var y2: Scalar[Self.dtype]
    var x2: Scalar[Self.dtype]

    def __init__(
        out self,
        y1: Scalar[Self.dtype],
        x1: Scalar[Self.dtype],
        y2: Scalar[Self.dtype],
        x2: Scalar[Self.dtype],
    ):
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2


@inline(.always)
def _iou[
    dtype: DType
](a: BoxCoords[dtype], b: BoxCoords[dtype]) -> Scalar[dtype]:
    """Reference IoU: the CPU kernel's corner normalization, no union guard."""
    var a_nw_y = max(a.y1, a.y2)
    var a_nw_x = max(a.x1, a.x2)
    var a_se_y = min(a.y1, a.y2)
    var a_se_x = min(a.x1, a.x2)
    var b_nw_y = max(b.y1, b.y2)
    var b_nw_x = max(b.x1, b.x2)
    var b_se_y = min(b.y1, b.y2)
    var b_se_x = min(b.x1, b.x2)
    var i_nw_y = min(a_nw_y, b_nw_y)
    var i_nw_x = min(a_nw_x, b_nw_x)
    var i_se_y = max(a_se_y, b_se_y)
    var i_se_x = max(a_se_x, b_se_x)
    var intersects = not (i_nw_x < i_se_x or i_nw_y < i_se_y)
    var inter = (i_se_y - i_nw_y) * (i_se_x - i_nw_x) if intersects else Scalar[
        dtype
    ](0)
    var union_ = (
        (a_se_y - a_nw_y) * (a_se_x - a_nw_x)
        + (b_se_y - b_nw_y) * (b_se_x - b_nw_x)
        - inter
    )
    return abs(inter) / abs(union_)


def _reference[
    dtype: DType
](
    boxes_list: List[BoxCoords[dtype]],
    scores_list: List[Scalar[dtype]],
    batch: Int,
    num_classes: Int,
    n: Int,
    max_output: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> List[Int64]:
    """Serial greedy NMS with the pinned tie order.

    Returns flat [b, c, box] rows in pair order, matching the kernel's
    output packing.
    """
    var rows = List[Int64](capacity=batch * num_classes * max_output * 3)
    var iou_t = iou_threshold.cast[dtype]()
    var score_t = score_threshold.cast[dtype]()
    for b in range(batch):
        for c in range(num_classes):
            # Pair scores are a strided slice of scores_list.
            var base = (b * num_classes + c) * n
            # Eligible indices, sorted score descending / index
            # ascending (insertion sort: n is small in tests).
            var order = List[Int](capacity=n)
            for j in range(n):
                if scores_list[base + j] > score_t:
                    order.append(j)
            var i = 1
            while i < len(order):
                var jx = i
                while jx > 0:
                    var a = order[jx - 1]
                    var bj = order[jx]
                    var swap = False
                    if scores_list[base + a] < scores_list[base + bj]:
                        swap = True
                    elif (
                        scores_list[base + a] == scores_list[base + bj]
                        and bj < a
                    ):
                        swap = True
                    if swap:
                        order[jx - 1] = bj
                        order[jx] = a
                    else:
                        break
                    jx -= 1
                i += 1
            # Greedy.
            var alive = List[Bool](length=n, fill=True)
            var selected = 0
            var k = 0
            while k < len(order) and selected < max_output:
                var j = order[k]
                if not alive[j]:
                    k += 1
                    continue
                alive[j] = False
                rows.append(Int64(b))
                rows.append(Int64(c))
                rows.append(Int64(j))
                selected += 1
                for m in range(k + 1, len(order)):
                    var j2 = order[m]
                    if alive[j2]:
                        if (
                            _iou[dtype](
                                boxes_list[b * n + j], boxes_list[b * n + j2]
                            )
                            > iou_t
                        ):
                            alive[j2] = False
                k += 1
    return rows^


def _run_case[
    dtype: DType
](
    ctx: DeviceContext,
    batch: Int,
    num_classes: Int,
    n: Int,
    boxes_list: List[BoxCoords[dtype]],
    scores_list: List[Scalar[dtype]],
    max_output: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    label: String,
) raises:
    var expected = _reference[dtype](
        boxes_list,
        scores_list,
        batch,
        num_classes,
        n,
        max_output,
        iou_threshold,
        score_threshold,
    )
    var expected_rows = len(expected) // 3

    # --- CPU path ---
    var boxes_cpu = ctx.enqueue_create_host_buffer[dtype](batch * n * 4)
    var scores_cpu = ctx.enqueue_create_host_buffer[dtype](
        batch * num_classes * n
    )
    for i in range(batch * n):
        boxes_cpu[4 * i] = boxes_list[i].y1
        boxes_cpu[4 * i + 1] = boxes_list[i].x1
        boxes_cpu[4 * i + 2] = boxes_list[i].y2
        boxes_cpu[4 * i + 3] = boxes_list[i].x2
    for i in range(batch * num_classes * n):
        scores_cpu[i] = scores_list[i]

    var boxes_cpu_t = TileTensor(boxes_cpu, row_major(Coord(batch, n, 4)))
    var scores_cpu_t = TileTensor(
        scores_cpu, row_major(Coord(batch, num_classes, n))
    )

    var cpu_rows_ptr = ctx.enqueue_create_host_buffer[.int64](
        expected_rows * 3 + _GUARD
    )
    for g in range(_GUARD):
        cpu_rows_ptr[expected_rows * 3 + g] = Int64(-12345)
    var cpu_out_t = TileTensor(cpu_rows_ptr, row_major(Coord(expected_rows, 3)))
    non_max_suppression[dtype](
        boxes_cpu_t,
        scores_cpu_t,
        cpu_out_t,
        max_output,
        iou_threshold,
        score_threshold,
    )

    # --- GPU path ---
    var boxes_dev = ctx.enqueue_create_buffer[dtype](batch * n * 4)
    var scores_dev = ctx.enqueue_create_buffer[dtype](batch * num_classes * n)
    var out_dev = ctx.enqueue_create_buffer[.int64](expected_rows * 3 + _GUARD)
    ctx.enqueue_copy(boxes_dev, boxes_cpu)
    ctx.enqueue_copy(scores_dev, scores_cpu)

    var boxes_dev_t = TileTensor(boxes_dev, row_major(Coord(batch, n, 4)))
    var scores_dev_t = TileTensor(
        scores_dev, row_major(Coord(batch, num_classes, n))
    )
    var out_dev_t = TileTensor(out_dev, row_major(Coord(expected_rows, 3)))
    # Poison the whole device allocation before running: a row the
    # kernel fails to write, or a write past the row count, is visible.
    var poison_host = ctx.enqueue_create_host_buffer[.int64](
        expected_rows * 3 + _GUARD
    )
    for g in range(expected_rows * 3 + _GUARD):
        poison_host[g] = Int64(-12345)
    ctx.enqueue_copy(out_dev, poison_host)
    ctx.synchronize()

    non_max_suppression[target="gpu"](
        boxes_dev_t.as_imm(),
        scores_dev_t.as_imm(),
        out_dev_t,
        max_output,
        iou_threshold,
        score_threshold,
        ctx,
    )

    var gpu_rows_ptr = ctx.enqueue_create_host_buffer[.int64](
        expected_rows * 3 + _GUARD
    )
    for g in range(expected_rows * 3 + _GUARD):
        gpu_rows_ptr[g] = Int64(-12345)
    ctx.enqueue_copy(gpu_rows_ptr, out_dev)
    ctx.synchronize()
    # GPU must match the pinned-order reference exactly.
    for r in range(expected_rows):
        assert_equal(
            Int(gpu_rows_ptr[3 * r]),
            Int(expected[3 * r]),
            msg=String(t"{label}: gpu row {r} batch"),
        )
        assert_equal(
            Int(gpu_rows_ptr[3 * r + 1]),
            Int(expected[3 * r + 1]),
            msg=String(t"{label}: gpu row {r} class"),
        )
        assert_equal(
            Int(gpu_rows_ptr[3 * r + 2]),
            Int(expected[3 * r + 2]),
            msg=String(t"{label}: gpu row {r} box"),
        )

    # Guard band: nothing past the row count may change.
    for g in range(_GUARD):
        assert_equal(
            Int(gpu_rows_ptr[expected_rows * 3 + g]),
            -12345,
            msg=String(t"{label}: gpu guard band"),
        )
    # CPU must match: both paths pin the same total order.
    for r in range(expected_rows):
        assert_equal(
            Int(cpu_rows_ptr[3 * r + 2]),
            Int(expected[3 * r + 2]),
            msg=String(t"{label}: cpu row {r} box"),
        )

    # The CPU guard band must be untouched too.
    for g in range(_GUARD):
        assert_equal(
            Int(cpu_rows_ptr[expected_rows * 3 + g]),
            -12345,
            msg=String(t"{label}: cpu guard band"),
        )

    _ = boxes_dev^
    _ = scores_dev^
    _ = out_dev^


@inline(.always)
def _box[dtype: DType](y: Int, x: Int, size: Int = 4) -> BoxCoords[dtype]:
    """Box at integer grid (y, x) spanning `size` units a side.

    Same-row neighbors at offset 1 have IoU 4*3/(2*16-12) = 0.6 (they
    suppress each other at threshold 0.5); offset 2 gives 1/3 and
    offset 3 gives 1/7 (both survive).
    """
    return BoxCoords[dtype](
        Scalar[dtype](y),
        Scalar[dtype](x),
        Scalar[dtype](y + size),
        Scalar[dtype](x + size),
    )


def _grid_boxes[
    dtype: DType
](batch: Int, n: Int, cols: Int = 8) -> List[BoxCoords[dtype]]:
    """Boxes on a grid: box j at (j // cols, j % cols), 4 units a side.

    Overlaps chains within rows and across rows, so greedy suppression
    has real work: same-row offset-1 pairs suppress (IoU 0.6), larger
    offsets survive.
    """
    var out = List[BoxCoords[dtype]](capacity=batch * n)
    for i in range(batch * n):
        var j = i % n
        out.append(_box[dtype](j // cols, j % cols))
    return out^


def _distinct_scores[
    dtype: DType
](batch: Int, num_classes: Int, n: Int, salt: Int) -> List[Scalar[dtype]]:
    """Distinct scores per (b, c) pair; pairs differ so packing is observable.
    """
    var out = List[Scalar[dtype]](capacity=batch * num_classes * n)
    for b in range(batch):
        for c in range(num_classes):
            for j in range(n):
                out.append(
                    Scalar[dtype](
                        Scalar[dtype](n - j)
                        + Scalar[dtype]((b * 7 + c * 11 + salt) % 5) * 100
                    )
                )
    return out^


def test_basic_sizes(ctx: DeviceContext) raises:
    comptime f32 = DType.float32
    # Sweep n across bitonic stage boundaries and pair shapes.
    for n in [1, 2, 3, 7, 8, 31, 32, 33, 64, 255, 256, 257]:
        _run_case[f32](
            ctx,
            1,
            1,
            n,
            _grid_boxes[f32](1, n),
            _distinct_scores[f32](1, 1, n, 1),
            10,
            0.5,
            0.0,
            String(t"basic n={n}"),
        )


def test_batch_and_classes(ctx: DeviceContext) raises:
    comptime f32 = DType.float32
    # Rows must pack sequentially in pair order across (b, c).
    _run_case[f32](
        ctx,
        3,
        4,
        9,
        _grid_boxes[f32](3, 9),
        _distinct_scores[f32](3, 4, 9, 2),
        2,
        0.5,
        0.0,
        "batch 3 classes 4",
    )


def test_thresholds(ctx: DeviceContext) raises:
    comptime f32 = DType.float32
    # Strict >: score == threshold is filtered; IoU == threshold survives.
    var boxes = List[BoxCoords[f32]](capacity=2)
    boxes.append(_box[f32](0, 0))
    boxes.append(_box[f32](0, 0))  # identical: IoU == 1
    var scores = List[Scalar[f32]](capacity=2)
    scores.append(Scalar[f32](0.5))
    scores.append(Scalar[f32](0.4))
    _run_case[f32](
        ctx, 1, 1, 2, boxes, scores, 5, 0.99, 0.5, "strict score threshold"
    )

    var boxes2 = List[BoxCoords[f32]](capacity=2)
    boxes2.append(_box[f32](0, 0))
    boxes2.append(_box[f32](0, 0))
    var scores2 = List[Scalar[f32]](capacity=2)
    scores2.append(Scalar[f32](0.9))
    scores2.append(Scalar[f32](0.8))
    _run_case[f32](
        ctx, 1, 1, 2, boxes2, scores2, 5, 1.0, 0.0, "iou == survives"
    )


def test_max_output_zero(ctx: DeviceContext) raises:
    comptime f32 = DType.float32
    _run_case[f32](
        ctx,
        1,
        1,
        4,
        _grid_boxes[f32](1, 4),
        _distinct_scores[f32](1, 1, 4, 3),
        0,
        0.5,
        0.0,
        "k=0",
    )


def test_all_filtered(ctx: DeviceContext) raises:
    comptime f32 = DType.float32
    _run_case[f32](
        ctx,
        1,
        1,
        4,
        _grid_boxes[f32](1, 4),
        _distinct_scores[f32](1, 1, 4, 1),
        5,
        0.5,
        10000.0,
        "all filtered",
    )


def test_ties_pinned_order(ctx: DeviceContext) raises:
    """Ties follow the pinned order (score desc, index asc) on both paths.

    The A-B-C case: A and C don't overlap, B overlaps both. A (index 0)
    wins the tie, B is suppressed, C survives. Both the CPU comparator
    and the GPU kernel pin the same total order, so the CPU result is
    compared too.
    """
    comptime f32 = DType.float32
    var boxes = List[BoxCoords[f32]](capacity=3)
    boxes.append(_box[f32](0, 0))
    boxes.append(_box[f32](0, 1))
    boxes.append(_box[f32](0, 2))
    var scores = List[Scalar[f32]](capacity=3)
    scores.append(Scalar[f32](0.5))
    scores.append(Scalar[f32](0.5))
    scores.append(Scalar[f32](0.5))
    _run_case[f32](
        ctx,
        1,
        1,
        3,
        boxes,
        scores,
        5,
        0.4,
        0.0,
        "ties abc pinned",
    )


def test_zero_area(ctx: DeviceContext) raises:
    """Zero-area boxes: union 0, IoU NaN; NaN > t is false, both survive."""
    comptime f32 = DType.float32
    var boxes = List[BoxCoords[f32]](capacity=2)
    boxes.append(BoxCoords[f32](1.0, 1.0, 1.0, 1.0))
    boxes.append(BoxCoords[f32](1.0, 1.0, 1.0, 1.0))
    var scores = List[Scalar[f32]](capacity=2)
    scores.append(Scalar[f32](0.9))
    scores.append(Scalar[f32](0.8))
    _run_case[f32](ctx, 1, 1, 2, boxes, scores, 5, 0.5, 0.0, "zero area")


def test_f64_dispatch_falls_back(ctx: DeviceContext) raises:
    """Float64 dispatches to the host fallback with device tensors.

    The fallback copies device -> host -> device around the CPU kernel;
    the result must still match the reference exactly.
    """
    comptime f64 = DType.float64
    _run_case[f64](
        ctx,
        2,
        3,
        5,
        _grid_boxes[f64](2, 5),
        _distinct_scores[f64](2, 3, 5, 1),
        4,
        0.5,
        0.0,
        "f64 dispatch fallback",
    )


def test_nan_scores(ctx: DeviceContext) raises:
    """NaN scores are ineligible and must not stall the comparator.

    A NaN fails the strict > threshold filter, so its box is dead; if the
    raw NaN stayed in the sort keys it would never swap and could push
    alive candidates past the winner scan's [0, n) range. The pinned
    order must still hold for the finite scores.
    """
    comptime f32 = DType.float32
    var boxes = _grid_boxes[f32](1, 6)
    var scores = List[Scalar[f32]](capacity=6)
    scores.append(Scalar[f32](0.3))
    scores.append(Scalar[f32](1.0))
    scores.append(Scalar[f32](0.3))
    scores.append(Scalar[f32](1.0))
    scores.append(Scalar[f32](0.9))
    scores.append(nan[f32]())
    _run_case[f32](ctx, 1, 1, 6, boxes, scores, 5, 0.5, 0.0, "nan score")


def test_large_n(ctx: DeviceContext) raises:
    comptime f32 = DType.float32
    _run_case[f32](
        ctx,
        1,
        1,
        2048,
        _grid_boxes[f32](1, 2048),
        _distinct_scores[f32](1, 1, 2048, 2),
        100,
        0.5,
        0.0,
        "large n=2048",
    )
    # The exact shared-memory budget: the largest GPU-path shape.
    _run_case[f32](
        ctx,
        1,
        1,
        4096,
        _grid_boxes[f32](1, 4096),
        _distinct_scores[f32](1, 1, 4096, 2),
        100,
        0.5,
        0.0,
        "budget n=4096",
    )


def test_over_budget_fallback(ctx: DeviceContext) raises:
    """A box count one past the budget dispatches to the host fallback."""
    comptime f32 = DType.float32
    _run_case[f32](
        ctx,
        1,
        1,
        4097,
        _grid_boxes[f32](1, 4097),
        _distinct_scores[f32](1, 1, 4097, 2),
        100,
        0.5,
        0.0,
        "over-budget n=4097",
    )


def main() raises:
    var ctx = DeviceContext()
    test_basic_sizes(ctx)
    test_batch_and_classes(ctx)
    test_thresholds(ctx)
    test_max_output_zero(ctx)
    test_all_filtered(ctx)
    test_ties_pinned_order(ctx)
    test_zero_area(ctx)
    test_f64_dispatch_falls_back(ctx)
    test_nan_scores(ctx)
    test_large_n(ctx)
    test_over_budget_fallback(ctx)
    print("All NMS GPU tests passed")
