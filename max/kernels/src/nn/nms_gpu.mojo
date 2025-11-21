# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from math import iota

from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout, RuntimeTuple

from utils import IndexList

from memory import UnsafePointer
from gpu import thread_idx, block_dim, block_idx, barrier
from gpu.host import DeviceContext, DeviceBuffer

@fieldwise_init
struct BoundingBox[dtype: DType](ImplicitlyCopyable, Movable):
    """Represents a 2D bounding box for object detection.

    The box is stored using two corner points: `nw` and `se`.
    **Note:** In this implementation, `nw` stores the maximum coordinates (max y, max x)
    and `se` stores the minimum coordinates (min y, min x). This differs from the typical
    interpretation of "northwest" (usually min x, max y) and "southeast" (usually max x, min y).
    This representation allows efficient computation of intersection and union areas.

    Parameters:
        dtype: The data type for coordinate values.

    Fields:
        nw: Corner storing the maximum coordinates (max y, max x).
        se: Corner storing the minimum coordinates (min y, min x).
    """

    var nw: SIMD[dtype, 2]
    var se: SIMD[dtype, 2]

    fn __init__(
        out self,
        y1: Scalar[dtype],
        x1: Scalar[dtype],
        y2: Scalar[dtype],
        x2: Scalar[dtype],
    ):
        """Initialize a bounding box from two diagonal corner coordinates.

        Args:
            y1: Y-coordinate of first corner.
            x1: X-coordinate of first corner.
            y2: Y-coordinate of second corner.
            x2: X-coordinate of second corner.

        Note:
            The corners are automatically ordered to ensure nw contains the
            maximum coordinates and se contains the minimum coordinates.
        """
        self.nw = SIMD[dtype, 2](max(y1, y2), max(x1, x2))
        self.se = SIMD[dtype, 2](min(y1, y2), min(x1, x2))

    fn iou(self, other: BoundingBox[dtype]) -> Scalar[dtype]:
        """Calculate Intersection over Union (IoU) with another bounding box.

        Args:
            other: The other bounding box to compare with.

        Returns:
            The IoU value, ranging from 0 (no overlap) to 1 (perfect overlap).
        """
        var intersection_area = self.intersection_area(other)

        var union_area = self.area() + other.area() - intersection_area
        var iou_val = abs(intersection_area) / abs(union_area)
        return iou_val

    fn intersection_area(self, other: BoundingBox[dtype]) -> Scalar[dtype]:
        """Calculate the area of intersection with another bounding box.

        Args:
            other: The other bounding box to intersect with.

        Returns:
            The intersection area, or 0 if boxes don't overlap.
        """
        var nw = min(self.nw, other.nw)
        var se = max(self.se, other.se)

        # Check if boxes don't overlap (invalid intersection)
        if nw[1] < se[1] or nw[0] < se[0]:
            return 0

        return Self(nw, se).area()

    fn area(self) -> Scalar[dtype]:
        """Calculate the area of this bounding box.

        Returns:
            The area of the box.
        """
        return (self.se[0] - self.nw[0]) * (self.se[1] - self.nw[1])


@always_inline
fn _get_bounding_box[
    dtype: DType
](
    batch_size: Int,
    box_idx: Int,
    boxes: LayoutTensor[dtype, **_],
) -> BoundingBox[dtype]:
    """Extract a bounding box from a tensor of boxes.

    Args:
        batch_size: The batch index to extract from.
        box_idx: The box index within the batch.
        boxes: A rank-3 tensor containing boxes with shape (batch, num_boxes, 4).
               The last dimension contains [y1, x1, y2, x2] coordinates.

    Returns:
        A BoundingBox instance constructed from the extracted coordinates.
    """
    constrained[boxes.rank == 3, "boxes must be of rank 3"]()
    var y1 = boxes[batch_size, box_idx, 0][0]
    var x1 = boxes[batch_size, box_idx, 1][0]
    var y2 = boxes[batch_size, box_idx, 2][0]
    var x2 = boxes[batch_size, box_idx, 3][0]
    return BoundingBox(y1, x1, y2, x2)

fn non_max_suppression_kernel[
    layout: Layout, dtype: DType
](
    boxes: LayoutTensor[mut=False, dtype, layout],
    scores: LayoutTensor[mut=False, dtype, layout],
    batch_size: Int,
    num_classes: Int,
    num_boxes: Int,
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    out_ptr: UnsafePointer[Int64],
):
    # Flatten (b, c) into 1D thread index
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total = batch_size * num_classes

    if tid >= total:
        return

    var b:Int = Int(tid / num_classes)
    var c = tid % num_classes
    
    if max_output_boxes_per_class == 0:
        return

    # Allocate local buffers (per thread)
    var box_idxs = List[Int64](unsafe_uninit_length=num_boxes)
    var per_class_scores = List[Scalar[dtype]](unsafe_uninit_length=num_boxes)

    # Get pointer to scores[b, c, 0]
    var offset = (b * num_classes + c) * num_boxes
    var per_class_scores_ptr = scores.ptr + offset

    # Filter by score threshold
    var num_boxes_remaining = 0
    for i in range(num_boxes):
        var score = per_class_scores_ptr.load(i)
        if score > score_threshold.cast[dtype]():
            per_class_scores[i] = score
            num_boxes_remaining += 1
        else:
            per_class_scores[i] = Scalar[dtype].MIN  # ~ -inf

    # Initialize box indices [0, 1, 2, ..., num_boxes-1]
    iota(box_idxs)

    @parameter
    @always_inline
    fn _greater_than(lhs: Int64, rhs: Int64) -> Bool:
        """Compare boxes by their scores in descending order."""
        return per_class_scores[Int(lhs)] > per_class_scores[Int(rhs)]

    # Sort box indices by descending score
    sort[_greater_than](box_idxs)

    # This thread's reserved slice in the output:
    # each (b,c) can produce at most max_output_boxes_per_class boxes.
    # Global row index for (b,c,k):
    #   row = ((b * num_classes) + c) * max_output_boxes_per_class + k
    var base_row: Int64 = (
        Int64(b * num_classes + c) * Int64(max_output_boxes_per_class)
    )
    var written: Int64 = 0
    var pred_idx = 0
    while (
        pred_idx < max_output_boxes_per_class
        and num_boxes_remaining > 0
    ):
        # Select highest-scoring remaining box
        var pred = _get_bounding_box(b, Int(box_idxs[pred_idx]), boxes)
        num_boxes_remaining -= 1

        # Emit output if we still have room in this (b,c) slice
        if written < Int64(max_output_boxes_per_class):
            var row = base_row + written
            var base = row * 3  # [batch, class, box]
            (out_ptr + base + 0).store(Int64(b))
            (out_ptr + base + 1).store(Int64(c))
            (out_ptr + base + 2).store(box_idxs[pred_idx])
            written += 1

        # At this point box_idxs is sorted like:
        # [best, 2nd best, ..., num_boxes_remaining'th best, -inf, ...]
        var num_boxes_curr_pred = num_boxes_remaining

        # Suppress overlaps
        for i in range(pred_idx + 1, pred_idx + 1 + num_boxes_curr_pred):
            var next_box = _get_bounding_box(b, Int(box_idxs[i]), boxes)

            if pred.iou(next_box) > iou_threshold.cast[dtype]():
                per_class_scores[Int(box_idxs[i])] = Scalar[dtype].MIN
                num_boxes_remaining -= 1

        pred_idx += 1

        # Re-sort remaining candidates (same trick as CPU)
        sort[_greater_than](
            Span[box_idxs.T, origin_of(box_idxs)](
                ptr=box_idxs.unsafe_ptr() + pred_idx,
                length=UInt(num_boxes_curr_pred),
            )
        )

fn non_max_suppression_gpu[
    dtype: DType
](
    boxes_dev: DeviceBuffer[dtype],
    scores_dev: DeviceBuffer[dtype],
    out_dev: DeviceBuffer[DType.int64],
    batch_size: Int,
    num_classes: Int,
    num_boxes: Int,
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    ctx: DeviceContext,
) raises:

    alias nms_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
    # Build runtime layouts for boxes [B, N, 4] and scores [B, C, N]
    var boxes_shape = IndexList[3](batch_size, num_boxes, 4)
    var scores_shape = IndexList[3](batch_size, num_classes, num_boxes)

    var boxes = LayoutTensor[mut=False, dtype, nms_layout](
        boxes_dev.unsafe_ptr(),
        RuntimeLayout[nms_layout].row_major(boxes_shape),
    )

    var scores = LayoutTensor[mut=False, dtype, nms_layout](
        scores_dev.unsafe_ptr(),
        RuntimeLayout[nms_layout].row_major(scores_shape),
    )

    # Launch 1D grid: one thread per (b,c)
    var total = batch_size * num_classes
    var threads_per_block = 128
    var num_blocks = (total + threads_per_block - 1) // threads_per_block

    ctx.enqueue_function[
        non_max_suppression_kernel[nms_layout, dtype]
    ](
        boxes,
        scores,
        batch_size,
        num_classes,
        num_boxes,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        out_dev.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(threads_per_block, 1, 1),
    )
