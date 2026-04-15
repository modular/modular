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

"""Mojo kernel wrappers for non-maximum suppression (NMS) interpreter op.

NMS filters out boxes with high intersection-over-union (IoU). The output
size is data-dependent (the number of surviving boxes varies with input
data). The Python handler pre-allocates an upper-bound buffer
(``batch * classes * max_output_per_class``) and a 1-element count buffer,
then calls ``NmsRun`` which fills both in a single pass — avoiding a
redundant O(n²) re-computation.

CPU-only: ``mo.non_maximum_suppression`` carries the ``MO_HostOnly`` trait.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import OpaquePointer

from op_utils import (
    _get_dtype,
    _make_ptr,
    Dispatchable,
    dispatch_dtype,
)


@export
def PyInit_nms_ops() -> PythonObject:
    """Create a Python module with NMS kernel function bindings."""
    try:
        var b = PythonModuleBuilder("nms_ops")
        b.def_function[nms_run_dispatcher](
            "NmsRun",
            docstring=(
                "Run NMS in a single pass: fills an upper-bound output buffer"
                " and writes the actual count"
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create nms_ops bindings module: {e}")


# ===----------------------------------------------------------------------=== #
# IoU helper
# ===----------------------------------------------------------------------=== #


@always_inline
def _iou[
    dtype: DType, //
](
    boxes_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    stride: Int,
    i: Int,
    j: Int,
) -> Float64:
    """Compute IoU between boxes[i] and boxes[j].

    Boxes are stored as [y1, x1, y2, x2] with `stride` elements between
    consecutive boxes (i.e. boxes[i] starts at boxes_ptr[i * stride]).

    Parameters:
        dtype: Element dtype of the boxes buffer.

    Args:
        boxes_ptr: Pointer to the flat boxes buffer for one batch.
        stride: Number of elements between consecutive boxes (always 4).
        i: Index of the first box.
        j: Index of the second box.

    Returns:
        IoU value in [0.0, 1.0].
    """
    var y1_i = Float64(boxes_ptr[i * stride + 0])
    var x1_i = Float64(boxes_ptr[i * stride + 1])
    var y2_i = Float64(boxes_ptr[i * stride + 2])
    var x2_i = Float64(boxes_ptr[i * stride + 3])

    var y1_j = Float64(boxes_ptr[j * stride + 0])
    var x1_j = Float64(boxes_ptr[j * stride + 1])
    var y2_j = Float64(boxes_ptr[j * stride + 2])
    var x2_j = Float64(boxes_ptr[j * stride + 3])

    # Normalize so (y1, x1) is top-left and (y2, x2) is bottom-right.
    var ay1 = min(y1_i, y2_i)
    var ay2 = max(y1_i, y2_i)
    var ax1 = min(x1_i, x2_i)
    var ax2 = max(x1_i, x2_i)

    var by1 = min(y1_j, y2_j)
    var by2 = max(y1_j, y2_j)
    var bx1 = min(x1_j, x2_j)
    var bx2 = max(x1_j, x2_j)

    var inter_y1 = max(ay1, by1)
    var inter_x1 = max(ax1, bx1)
    var inter_y2 = min(ay2, by2)
    var inter_x2 = min(ax2, bx2)

    var inter_area = max(Float64(0), inter_y2 - inter_y1) * max(
        Float64(0), inter_x2 - inter_x1
    )
    var area_a = (ay2 - ay1) * (ax2 - ax1)
    var area_b = (by2 - by1) * (bx2 - bx1)
    var union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ===----------------------------------------------------------------------=== #
# Greedy NMS core
# ===----------------------------------------------------------------------=== #


@always_inline
def _greedy_nms[
    dtype: DType, //
](
    boxes_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scores_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    num_boxes: Int,
    max_output: Int,
    iou_thresh: Float64,
    score_thresh: Float64,
) -> List[Int]:
    """Run greedy NMS on a single (batch, class) slice.

    Parameters:
        dtype: Element dtype of the boxes/scores buffers.

    Args:
        boxes_ptr: Pointer to boxes for this batch, shape [num_boxes, 4].
        scores_ptr: Pointer to scores for this (batch, class), length num_boxes.
        num_boxes: Number of boxes.
        max_output: Maximum boxes to select.
        iou_thresh: IoU suppression threshold.
        score_thresh: Score threshold.

    Returns:
        List of selected box indices (original indices into the input).
    """
    var sorted_indices = List[Int](unsafe_uninit_length=num_boxes)
    var suppressed = List[Bool](unsafe_uninit_length=num_boxes)

    # Build list of indices passing score threshold.
    var n_candidates = 0
    for i in range(num_boxes):
        if Float64(scores_ptr[i]) > score_thresh:
            sorted_indices[n_candidates] = i
            n_candidates += 1

    # Insertion sort by score descending.  O(n²) worst-case, but NMS
    # itself is O(n²) so the sort doesn't dominate.  Insertion sort's low
    # overhead is a win for typical detector box counts (≤ ~10K per class).
    for i in range(1, n_candidates):
        var key_idx = sorted_indices[i]
        var key_score = Float64(scores_ptr[key_idx])
        var j = i - 1
        while j >= 0 and Float64(scores_ptr[sorted_indices[j]]) < key_score:
            sorted_indices[j + 1] = sorted_indices[j]
            j -= 1
        sorted_indices[j + 1] = key_idx

    # Clear suppression flags.
    for i in range(n_candidates):
        suppressed[i] = False

    # Greedy selection.
    var selected = List[Int]()
    for i in range(n_candidates):
        if suppressed[i]:
            continue
        if len(selected) >= max_output:
            break
        var idx = sorted_indices[i]
        selected.append(idx)
        # Suppress all remaining boxes with IoU > threshold.
        for j in range(i + 1, n_candidates):
            if not suppressed[j]:
                var other_idx = sorted_indices[j]
                if _iou(boxes_ptr, 4, idx, other_idx) > iou_thresh:
                    suppressed[j] = True

    return selected^


# ===----------------------------------------------------------------------=== #
# Single-pass NMS kernel
# ===----------------------------------------------------------------------=== #


@always_inline
def nms_run_op[
    dtype: DType, //
](
    count_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[DType.int64], MutExternalOrigin],
    boxes_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scores_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    batch_size: Int,
    num_classes: Int,
    num_boxes: Int,
    max_output: Int,
    iou_thresh: Float64,
    score_thresh: Float64,
):
    """Run greedy NMS and fill results in a single pass.

    Writes [batch_index, class_index, box_index] rows into ``out_ptr``
    (pre-allocated to the upper bound ``batch * classes * max_output``)
    and writes the actual count to ``count_ptr[0]``.

    Parameters:
        dtype: Element dtype of the boxes/scores buffers.

    Args:
        count_ptr: 1-element int64 buffer; receives the total selected count.
        out_ptr: Upper-bound int64 buffer of shape [upper_bound, 3].
        boxes_ptr: Flat boxes buffer [batch, num_boxes, 4].
        scores_ptr: Flat scores buffer [batch, num_classes, num_boxes].
        batch_size: Batch dimension.
        num_classes: Number of classes.
        num_boxes: Number of boxes per batch.
        max_output: Maximum output boxes per class.
        iou_thresh: IoU suppression threshold.
        score_thresh: Score threshold.
    """
    var row = 0
    for b in range(batch_size):
        var batch_boxes = boxes_ptr + b * num_boxes * 4
        for c in range(num_classes):
            var class_scores = scores_ptr + (b * num_classes + c) * num_boxes
            var selected = _greedy_nms(
                batch_boxes,
                class_scores,
                num_boxes,
                max_output,
                iou_thresh,
                score_thresh,
            )
            for i in range(len(selected)):
                out_ptr[row * 3 + 0] = Int64(b)
                out_ptr[row * 3 + 1] = Int64(c)
                out_ptr[row * 3 + 2] = Int64(selected[i])
                row += 1

    count_ptr[0] = Int64(row)


@fieldwise_init
struct _NmsRunBody(Dispatchable):
    """Dispatch body for single-pass NMS over float dtypes."""

    var count_addr: Int
    var out_addr: Int
    var boxes_addr: Int
    var scores_addr: Int
    var batch_size: Int
    var num_classes: Int
    var num_boxes: Int
    var max_output: Int
    var iou_thresh: Float64
    var score_thresh: Float64

    def call[t: DType](self) raises -> None:
        nms_run_op(
            _make_ptr[DType.int64](self.count_addr),
            _make_ptr[DType.int64](self.out_addr),
            _make_ptr[t](self.boxes_addr),
            _make_ptr[t](self.scores_addr),
            self.batch_size,
            self.num_classes,
            self.num_boxes,
            self.max_output,
            self.iou_thresh,
            self.score_thresh,
        )


def nms_run_dispatcher(
    count_buffer: PythonObject,
    out_buffer: PythonObject,
    boxes_buffer: PythonObject,
    scores_buffer: PythonObject,
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """NmsRun dispatcher: single-pass NMS into upper-bound buffer.

    Args:
        count_buffer: Pre-allocated 1-element int64 buffer for result count.
        out_buffer: Pre-allocated upper-bound int64 buffer [upper_bound, 3].
        boxes_buffer: Input boxes buffer [batch, num_boxes, 4].
        scores_buffer: Input scores buffer [batch, num_classes, num_boxes].
        params: Python tuple (batch_size, num_classes, num_boxes,
                max_output_boxes_per_class, iou_threshold, score_threshold).
        device_context_ptr: Device context pointer (unused; CPU-only).
    """
    var dtype = _get_dtype(boxes_buffer)
    dispatch_dtype(
        _NmsRunBody(
            Int(py=count_buffer._data_ptr()),
            Int(py=out_buffer._data_ptr()),
            Int(py=boxes_buffer._data_ptr()),
            Int(py=scores_buffer._data_ptr()),
            Int(py=params[0]),
            Int(py=params[1]),
            Int(py=params[2]),
            Int(py=params[3]),
            Float64(py=params[4]),
            Float64(py=params[5]),
        ),
        dtype,
    )
