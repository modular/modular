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
"""Implements non-maximum suppression (NMS) for object detection bounding-box filtering."""

from std.math import iota

from std.memory import unsafe_stack_allocation

from layout import (
    DefaultEngine,
    Idx,
    ImmTileTensor,
    MutTileTensor,
    TensorEngine,
    TensorLayout,
    TileTensor,
)

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from max.gpu.sync import barrier

from std.utils import IndexList
from std.utils.index import StaticTuple


@fieldwise_init
struct BoundingBox[dtype: DType](ImplicitlyCopyable):
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

    var nw: SIMD[Self.dtype, 2]
    var se: SIMD[Self.dtype, 2]

    def __init__(
        out self,
        y1: Scalar[Self.dtype],
        x1: Scalar[Self.dtype],
        y2: Scalar[Self.dtype],
        x2: Scalar[Self.dtype],
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
        self.nw = SIMD[Self.dtype, 2](max(y1, y2), max(x1, x2))
        self.se = SIMD[Self.dtype, 2](min(y1, y2), min(x1, x2))

    def iou(self, other: BoundingBox[Self.dtype]) -> Scalar[Self.dtype]:
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

    def intersection_area(
        self, other: BoundingBox[Self.dtype]
    ) -> Scalar[Self.dtype]:
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

    def area(self) -> Scalar[Self.dtype]:
        """Calculate the area of this bounding box.

        Returns:
            The area of the box.
        """
        return (self.se[0] - self.nw[0]) * (self.se[1] - self.nw[1])


@inline(.always)
def _get_bounding_box[
    dtype: DType
](
    batch_size: Int,
    box_idx: Int,
    boxes: TileTensor[dtype, ...],
) -> BoundingBox[
    dtype
]:
    """Extract a bounding box from a tensor of boxes.

    Args:
        batch_size: The batch index to extract from.
        box_idx: The box index within the batch.
        boxes: A rank-3 tensor containing boxes with shape (batch, num_boxes, 4).
               The last dimension contains [y1, x1, y2, x2] coordinates.

    Returns:
        A BoundingBox instance constructed from the extracted coordinates.
    """
    comptime assert boxes.flat_rank == 3, "boxes must be of rank 3"
    comptime assert boxes.element_size == 1

    var y1 = boxes[batch_size, box_idx, 0][0]
    var x1 = boxes[batch_size, box_idx, 1][0]
    var y2 = boxes[batch_size, box_idx, 2][0]
    var x2 = boxes[batch_size, box_idx, 3][0]
    return BoundingBox(y1, x1, y2, x2)


def non_max_suppression[
    dtype: DType
](
    boxes: TileTensor[dtype, ...],
    scores: TileTensor[dtype, ...],
    output: TileTensor[mut=True, .int64, ...],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
):
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.

    This is a buffer semantic overload that writes results directly to an output tensor.
    NMS iteratively selects boxes with highest scores while suppressing nearby boxes
    with high overlap (IoU).

    Parameters:
        dtype: The data type for box coordinates and scores.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape (batch, num_boxes, 4).
               Each box is [y1, x1, y2, x2].
        scores: Rank-3 tensor of scores with shape (batch, num_classes, num_boxes).
        output: Rank-2 output tensor to store selected boxes as (N, 3) where each
                row is [batch_idx, class_idx, box_idx].
        max_output_boxes_per_class: Maximum number of boxes to select per class.
        iou_threshold: IoU threshold for suppression. Boxes with IoU > threshold
                       are suppressed.
        score_threshold: Minimum score threshold. Boxes with score < threshold
                        are filtered out.
    """
    comptime assert boxes.flat_rank == 3, "boxes must be of rank 3"
    comptime assert scores.flat_rank == 3, "scores must be of rank 3"
    comptime assert output.flat_rank == 2, "output must be of rank 2"

    var pred_count = 0

    @inline(.always)
    def store_to_outputs(
        batch_idx: Int64, class_idx: Int64, box_idx: Int64
    ) {output, mut pred_count}:
        """Store selected box indices to output tensor."""
        output[pred_count, 0] = batch_idx
        output[pred_count, 1] = class_idx
        output[pred_count, 2] = box_idx
        pred_count += 1

    non_max_suppression[dtype](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        store_to_outputs,
    )


def non_max_suppression_shape_func[
    dtype: DType
](
    boxes: TileTensor[dtype, ...],
    scores: TileTensor[dtype, ...],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> IndexList[2]:
    """Compute the output shape for NMS without allocating the output buffer.

    This function performs a dry-run of NMS to determine how many boxes will be
    selected, allowing proper output buffer allocation. Can be removed once the
    graph compiler supports value semantic kernels that allocate their own output.

    Parameters:
        dtype: The data type for box coordinates and scores.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape (batch, num_boxes, 4).
        scores: Rank-3 tensor of scores with shape (batch, num_classes, num_boxes).
        max_output_boxes_per_class: Maximum number of boxes to select per class.
        iou_threshold: IoU threshold for suppression.
        score_threshold: Minimum score threshold.

    Returns:
        A 2-element IndexList specifying the output shape (num_selected_boxes, 3).
    """
    comptime assert boxes.flat_rank == 3, "boxes must be of rank 3"
    comptime assert scores.flat_rank == 3, "scores must be of rank 3"

    var box_pred_count: Int64 = 0

    @inline(.always)
    def incr_pred_count(
        batch_idx: Int64, class_idx: Int64, box_idx: Int64
    ) {mut box_pred_count}:
        """Count selected boxes without storing them."""
        box_pred_count += 1

    non_max_suppression[dtype](
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        incr_pred_count,
    )

    return IndexList[2](Int(box_pred_count), 3)


def non_max_suppression[
    dtype: DType,
    FuncType: ImplicitlyCopyable & def(Int64, Int64, Int64) -> None,
](
    boxes: TileTensor[dtype, ...],
    scores: TileTensor[dtype, ...],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    func: FuncType,
):
    """Implements the NonMaxSuppression operator from the ONNX spec https://github.com/onnx/onnx/blob/main/docs/Operators.md#nonmaxsuppression.

    Parameters:
        dtype: The data type for box coordinates and scores.
        FuncType: Type of the `func` callback invoked for each selected box.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape (batch, num_boxes, 4).
            Each box is [y1, x1, y2, x2].
        scores: Rank-3 tensor of detection scores with shape (batch,
            num_classes, num_boxes).
        max_output_boxes_per_class: Maximum number of boxes to select per
            class.
        iou_threshold: IoU threshold for suppression. Boxes with IoU above
            this value are suppressed.
        score_threshold: Minimum score for a box to be considered. Boxes
            below this are filtered out.
        func: Callback invoked for each selected box with the batch index,
            class index, and box index.
    """
    comptime assert boxes.rank == 3, "boxes must be of rank 3"
    comptime assert scores.rank == 3, "scores must be of rank 3"

    var batch_size = Int(boxes.dim[0]())
    var num_boxes = Int(boxes.dim[1]())
    var num_classes = Int(scores.dim[1]())

    assert Int(boxes.dim[2]()) == 4, (
        "boxes must be specified with the 2D coords representing the"
        " diagonal corners"
    )
    assert Int(boxes.dim[0]()) == Int(
        scores.dim[0]()
    ), "dim 0 of boxes and scores must be equal"
    assert Int(boxes.dim[1]()) == Int(
        scores.dim[2]()
    ), "boxes and scores must contain the same number of boxes"

    if max_output_boxes_per_class == 0:
        return

    # Allocate the box indices and scores without initializing their elements.
    var box_idxs = List[Int64](unsafe_uninit_length=num_boxes)
    var per_class_scores = List[Scalar[dtype]](unsafe_uninit_length=num_boxes)

    for b in range(batch_size):
        for c in range(num_classes):
            # Entries of the per-class scores are set to neginf when they no longer
            # correspond to an eligible box
            # this happens when:
            #   1. Score does not meet score threshold (filtered out initially)
            #   2. IoU with an already-selected boc is above IOU threshold (suppressed)
            # Filter boxes by score threshold
            # This reduces the number of boxes to sort and process
            var num_boxes_remaining = 0
            for i in range(num_boxes):
                var score = rebind[Scalar[dtype]](
                    scores.load[invariant=False]((b, c, i))
                )
                if score > score_threshold.cast[dtype]():
                    per_class_scores[i] = score
                    num_boxes_remaining += 1
                else:
                    per_class_scores[i] = Scalar[dtype].MIN  # ~ -inf

            # Initialize box indices [0, 1, 2, ..., num_boxes-1]
            iota(box_idxs)

            @inline(.always)
            def _greater_than(
                lhs: Int64, rhs: Int64
            ) {per_class_scores} -> Bool:
                """Compare boxes by their scores in descending order.

                Ties break by ascending box index, a total order shared
                with the GPU kernel so both paths -- and the shape
                function that sizes the output -- always agree on the
                selection.
                """
                var lhs_score = per_class_scores[Int(lhs)]
                var rhs_score = per_class_scores[Int(rhs)]
                if lhs_score == rhs_score:
                    return lhs < rhs
                return lhs_score > rhs_score

            # Sort box indices by descending score
            sort(box_idxs, _greater_than)

            # Iteratively select boxes and suppress overlapping ones
            var pred_idx = 0
            while (
                pred_idx < max_output_boxes_per_class
                and num_boxes_remaining > 0
            ):
                # Select the highest-scoring remaining box
                var pred = _get_bounding_box(b, Int(box_idxs[pred_idx]), boxes)
                num_boxes_remaining -= 1

                # Each output prediction contains 3 values: [batch_index, class_index, box_index]
                func(Int64(b), Int64(c), box_idxs[pred_idx])

                # At the beginning of this loop box_idxs are sorted such that scores[box_idxs] looks like this:
                # [1st best score, 2nd best score, ..., num_boxes_remaining'th best score, -inf, ..., -inf]
                var num_boxes_curr_pred = num_boxes_remaining

                # Suppress boxes with high IoU overlap
                # Iterate over remaining candidate boxes and mark those with IoU > threshold
                # as suppressed by setting their scores to MIN
                for i in range(
                    pred_idx + 1, pred_idx + 1 + num_boxes_curr_pred
                ):
                    var next_box = _get_bounding_box(b, Int(box_idxs[i]), boxes)

                    if pred.iou(next_box) > iou_threshold.cast[dtype]():
                        per_class_scores[Int(box_idxs[i])] = Scalar[dtype].MIN
                        num_boxes_remaining -= 1

                pred_idx += 1

                # We don't need to sort the entire array because:
                #   1. The start contains already-selected boxes (order doesn't matter)
                #   2. The end contains suppressed boxes with score=-inf (order doesn't matter)
                # Note: Use num_boxes_curr_pred (not num_boxes_remaining) because it
                # represents the count before we marked boxes as suppressed above
                var box_idxs_ptr = box_idxs.unsafe_ptr()
                sort(
                    Span(
                        unsafe_ptr=box_idxs_ptr.unsafe_offset(pred_idx),
                        length=num_boxes_curr_pred,
                    ),
                    _greater_than,
                )

            @inline(.always)
            def sorted() {imm} -> Bool:
                for i in range(len(box_idxs) - 1):
                    if (
                        per_class_scores[Int(box_idxs[i])]
                        < per_class_scores[Int(box_idxs[i + 1])]
                    ):
                        return False
                return True

            assert sorted(), "NonMaxSuppression boxes not sorted correctly"


# ===-----------------------------------------------------------------------===#
# GPU NMS: one CTA, pairs in batch-major/class-major order, greedy in-block
# ===-----------------------------------------------------------------------===#


comptime _NMS_BLOCK_SIZE = 256
comptime _NMS_MAX_LOCAL = 4096


@inline(.always)
def _iou_pair[
    dtype: DType
](
    a_y1: Scalar[dtype],
    a_x1: Scalar[dtype],
    a_y2: Scalar[dtype],
    a_x2: Scalar[dtype],
    b_y1: Scalar[dtype],
    b_x1: Scalar[dtype],
    b_y2: Scalar[dtype],
    b_x2: Scalar[dtype],
) -> Scalar[dtype]:
    """Corner-normalized IoU: the CPU kernel's exact formula and order.

    No zero-union guard: a zero-area pair reproduces the CPU path's NaN,
    and NaN comparisons stay false on both paths.
    """
    # nw = (max y, max x); se = (min y, min x), per box.
    var a_nw_y = max(a_y1, a_y2)
    var a_nw_x = max(a_x1, a_x2)
    var a_se_y = min(a_y1, a_y2)
    var a_se_x = min(a_x1, a_x2)
    var b_nw_y = max(b_y1, b_y2)
    var b_nw_x = max(b_x1, b_x2)
    var b_se_y = min(b_y1, b_y2)
    var b_se_x = min(b_x1, b_x2)

    var i_nw_y = min(a_nw_y, b_nw_y)
    var i_nw_x = min(a_nw_x, b_nw_x)
    var i_se_y = max(a_se_y, b_se_y)
    var i_se_x = max(a_se_x, b_se_x)

    var intersects = not (i_nw_x < i_se_x or i_nw_y < i_se_y)
    var intersection = (i_se_y - i_nw_y) * (
        i_se_x - i_nw_x
    ) if intersects else Scalar[dtype](0)
    var area_a = (a_se_y - a_nw_y) * (a_se_x - a_nw_x)
    var area_b = (b_se_y - b_nw_y) * (b_se_x - b_nw_x)
    var union_ = area_a + area_b - intersection
    return abs(intersection) / abs(union_)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_NMS_BLOCK_SIZE))
)
@__name(t"nms_greedy_{dtype}")
def _nms_greedy_kernel[
    dtype: DType,
    BoxesLayout: TensorLayout,
    ScoresLayout: TensorLayout,
    OutputLayout: TensorLayout,
    BoxesEngine: TensorEngine = DefaultEngine[element_width=1],
    ScoresEngine: TensorEngine = DefaultEngine[element_width=1],
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    boxes_arg: ImmTileTensor[
        dtype, BoxesLayout, ImmutAnyOrigin, Engine=BoxesEngine
    ],
    scores_arg: ImmTileTensor[
        dtype, ScoresLayout, ImmutAnyOrigin, Engine=ScoresEngine
    ],
    output_arg: MutTileTensor[
        .int64, OutputLayout, MutAnyOrigin, Engine=OutputEngine
    ],
    num_boxes: Int32,
    num_classes: Int32,
    max_output_per_class: Int32,
    iou_threshold: Float32,
    score_threshold: Float32,
):
    """GPU NMS: one CTA walks pairs in batch-major, class-major order.

    Candidates for the current pair live in shared memory as
    (score, index) pairs, bitonic-sorted by score descending with index
    ascending as the pinned tie order. The greedy loop is wholly
    in-block: thread 0 takes the first surviving candidate, every
    thread sweeps the rest computing IoU against it in parallel, and
    suppressed candidates are marked dead. Rows pack sequentially in
    pair order, matching the CPU kernel's output, and sh_row carries
    the running row index across pairs.
    """
    comptime BLOCK = _NMS_BLOCK_SIZE
    comptime PAD = _NMS_MAX_LOCAL
    var tid = thread_idx.x
    var n = Int(num_boxes)
    var n_classes = Int(num_classes)
    var cap = Int(max_output_per_class)

    var sh_score = unsafe_stack_allocation[
        PAD, Scalar[dtype], address_space=.SHARED
    ]()
    var sh_index = unsafe_stack_allocation[PAD, Int32, address_space=.SHARED]()
    var sh_alive = unsafe_stack_allocation[PAD, Bool, address_space=.SHARED]()
    var sh_row = unsafe_stack_allocation[1, Int64, address_space=.SHARED]()

    if tid == 0:
        sh_row[0] = 0

    var pair = 0
    while pair < Int(scores_arg.dim[0]()) * n_classes:
        var b, c = divmod(pair, n_classes)

        # Load, filter, and pad the current pair's candidates.
        var i = tid
        while i < n:
            var score = rebind[Scalar[dtype]](
                scores_arg.load((Int32(b), Int32(c), i))
            )
            if score > score_threshold.cast[dtype]():
                # Eligible: keep the real score so the sort's total order
                # matches the CPU path.
                sh_score[i] = score
                sh_alive[i] = True
            else:
                # Ineligible (also NaN, which fails strict >): park at
                # the finite floor -MAX_FINITE. Unlike the CPU filter's
                # negative infinity, this finite key keeps the sort total
                # on every dtype; ineligible lanes are also marked dead,
                # so the key value never competes for selection.
                sh_score[i] = -Scalar[dtype].MAX_FINITE
                sh_alive[i] = False
            sh_index[i] = Int32(i)
            i += BLOCK
        i = n + tid
        while i < PAD:
            sh_score[i] = -Scalar[dtype].MAX_FINITE
            sh_index[i] = Int32(0x7FFFFFFF)
            sh_alive[i] = False
            i += BLOCK
        barrier()

        # Bitonic sort by score descending, index ascending. Runtime
        # stage loops (PAD is a compile-time power of two). Descending
        # blocks: swap when the lower lane's score is smaller (or equal
        # score with larger index); ascending blocks invert.
        var k = 2
        while k <= PAD:
            var j = k >> 1
            while j > 0:
                barrier()
                i = tid
                while i < PAD:
                    var partner = i ^ j
                    if partner > i:
                        var ascending_block = (i & k) == 0
                        var vi = sh_score[i]
                        var vj = sh_score[partner]
                        # true when the lower lane must swap up
                        var wrong: Bool
                        if vi == vj:
                            wrong = (
                                sh_index[i] > sh_index[partner]
                            ) if ascending_block else (
                                sh_index[i] < sh_index[partner]
                            )
                        else:
                            # Geometrically-ascending blocks hold
                            # score-descending runs: the lower lane
                            # swaps up when its score is smaller
                            # (or equal score and larger index);
                            # other blocks invert. The final stage is
                            # one geometrically-ascending block, so
                            # the output is score descending.
                            wrong = (vi > vj) if not ascending_block else (
                                vi < vj
                            )
                        if wrong:
                            var t_s = sh_score[i]
                            sh_score[i] = vj
                            sh_score[partner] = t_s
                            var t_i = sh_index[i]
                            sh_index[i] = sh_index[partner]
                            sh_index[partner] = t_i
                            var t_a = sh_alive[i]
                            sh_alive[i] = sh_alive[partner]
                            sh_alive[partner] = t_a
                    i += BLOCK
                j >>= 1
            k <<= 1
        barrier()

        # Greedy selection for this pair. Thread 0 scans for the first
        # alive candidate; alive lanes are sorted score-descending, so
        # the first alive lane is the winner. sh_index[0] doubles as
        # the winner-broadcast channel: safe because the first alive
        # lane, once picked, is marked dead, so later rounds never
        # read it as a candidate.
        var sel = 0
        while sel < cap:
            var winner: Int32 = -1
            if tid == 0:
                var j = 0
                while j < n:
                    if sh_alive[j]:
                        winner = sh_index[j]
                        sh_alive[j] = False
                        break
                    j += 1
                sh_index[0] = winner
            barrier()
            winner = sh_index[0]
            if winner < 0:
                break

            if tid == 0:
                # The host sizes output rows to the shape-function count,
                # and both paths pin the same tie order, so the counts
                # agree; the clamp guards against future drift so this
                # can never write past the allocation.
                var row = sh_row[0]
                var out_rows = Int64(output_arg.dim[0]())
                if row < out_rows:
                    output_arg.store((row, Idx[0]), Int64(b))
                    output_arg.store((row, Idx[1]), Int64(c))
                    output_arg.store((row, Idx[2]), Int64(winner))
                sh_row[0] = row + 1

            # Suppress remaining candidates that overlap the winner.
            var w_y1 = rebind[Scalar[dtype]](
                boxes_arg.load((Int32(b), winner, Idx[0]))
            )
            var w_x1 = rebind[Scalar[dtype]](
                boxes_arg.load((Int32(b), winner, Idx[1]))
            )
            var w_y2 = rebind[Scalar[dtype]](
                boxes_arg.load((Int32(b), winner, Idx[2]))
            )
            var w_x2 = rebind[Scalar[dtype]](
                boxes_arg.load((Int32(b), winner, Idx[3]))
            )
            i = tid
            while i < n:
                if sh_alive[i]:
                    var cand = sh_index[i]
                    if (
                        _iou_pair[dtype](
                            w_y1,
                            w_x1,
                            w_y2,
                            w_x2,
                            rebind[Scalar[dtype]](
                                boxes_arg.load((Int32(b), cand, Idx[0]))
                            ),
                            rebind[Scalar[dtype]](
                                boxes_arg.load((Int32(b), cand, Idx[1]))
                            ),
                            rebind[Scalar[dtype]](
                                boxes_arg.load((Int32(b), cand, Idx[2]))
                            ),
                            rebind[Scalar[dtype]](
                                boxes_arg.load((Int32(b), cand, Idx[3]))
                            ),
                        )
                        > iou_threshold.cast[dtype]()
                    ):
                        sh_alive[i] = False
                i += BLOCK
            barrier()
            sel += 1
        barrier()
        pair += 1


def non_max_suppression_gpu[
    dtype: DType,
    BoxesLayout: TensorLayout,
    ScoresLayout: TensorLayout,
    OutputLayout: TensorLayout,
    BoxesEngine: TensorEngine = DefaultEngine[element_width=1],
    ScoresEngine: TensorEngine = DefaultEngine[element_width=1],
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    boxes: ImmTileTensor[
        dtype, BoxesLayout, ImmutAnyOrigin, Engine=BoxesEngine
    ],
    scores: ImmTileTensor[
        dtype, ScoresLayout, ImmutAnyOrigin, Engine=ScoresEngine
    ],
    output: TileTensor[
        mut=True, .int64, OutputLayout, MutAnyOrigin, Engine=OutputEngine
    ],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    ctx: DeviceContext,
) raises:
    """Runs NMS on the GPU: pairs in batch-major order, greedy state
    in one CTA.

    The output's first dimension must be the host-sized selection count;
    rows pack sequentially in pair order, matching the CPU kernel.

    Parameters:
        dtype: The data type for box coordinates and scores.
        BoxesLayout: Layout type of the boxes tensor.
        ScoresLayout: Layout type of the scores tensor.
        OutputLayout: Layout type of the output tensor.
        BoxesEngine: Engine type of the boxes tensor.
        ScoresEngine: Engine type of the scores tensor.
        OutputEngine: Engine type of the output tensor.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape
            `(batch, num_boxes, 4)` where each box is `[y1, x1, y2, x2]`.
        scores: Rank-3 tensor of scores with shape
            `(batch, num_classes, num_boxes)`.
        output: Rank-2 output tensor of selected rows `[b, c, box]`.
        max_output_boxes_per_class: Maximum number of boxes to select
            per class.
        iou_threshold: IoU threshold for suppression.
        score_threshold: Minimum score for a box to be considered.
        ctx: Device context used to enqueue GPU work.

    Raises:
        Error: If ``num_boxes`` exceeds the shared-memory candidate
            budget; that shape runs on the host path instead.
    """
    comptime assert boxes.flat_rank == 3, "boxes must be rank 3"
    comptime assert scores.flat_rank == 3, "scores must be rank 3"
    comptime assert output.flat_rank == 2, "output must be rank 2"

    var batch = Int(boxes.dim[0]())
    var num_boxes = Int(boxes.dim[1]())
    var num_classes = Int(scores.dim[1]())

    # The kernel takes Int32 pair coordinates and extents; dimensions
    # beyond Int32 cannot be handled without wrapping (and would need
    # exabytes of scores), so reject them here rather than silently
    # mis-executing.
    if (
        batch > Int(Int32.MAX_FINITE)
        or num_boxes > Int(Int32.MAX_FINITE)
        or num_classes > Int(Int32.MAX_FINITE)
    ):
        raise Error(
            t"non_max_suppression: batch, num_boxes, and num_classes"
            t" must fit in int32 for the GPU path"
        )

    # Same invariants the CPU kernel asserts; the verifier does not
    # enforce shape consistency between boxes and scores.
    assert Int(boxes.dim[2]()) == 4, (
        "boxes must be specified with the 2D coords representing the"
        " diagonal corners"
    )
    assert Int(boxes.dim[0]()) == Int(
        scores.dim[0]()
    ), "dim 0 of boxes and scores must be equal"
    assert num_boxes == Int(
        scores.dim[2]()
    ), "boxes and scores must contain the same number of boxes"

    if num_boxes > _NMS_MAX_LOCAL:
        raise Error(
            t"non_max_suppression: num_boxes {num_boxes} exceeds the GPU"
            t" local budget {_NMS_MAX_LOCAL}"
        )
    if max_output_boxes_per_class <= 0:
        return

    # Clamp before the Int32 narrowing: an int64 operand of 2**31 or
    # more would wrap negative and silently select nothing while the
    # host-sized output expects the full count. A cap above the box
    # count is equivalent to the box count itself.
    var cap = min(max_output_boxes_per_class, num_boxes)

    comptime kernel = _nms_greedy_kernel[
        dtype=dtype,
        BoxesLayout=BoxesLayout,
        ScoresLayout=ScoresLayout,
        OutputLayout=OutputLayout,
        BoxesEngine=BoxesEngine,
        ScoresEngine=ScoresEngine,
        OutputEngine=OutputEngine,
    ]
    ctx.enqueue_function[kernel](
        boxes,
        scores,
        output,
        Int32(num_boxes),
        Int32(num_classes),
        Int32(cap),
        iou_threshold,
        score_threshold,
        block_dim=_NMS_BLOCK_SIZE,
        grid_dim=1,
    )


def non_max_suppression[
    dtype: DType,
    target: StaticString,
    BoxesLayout: TensorLayout,
    ScoresLayout: TensorLayout,
    OutputLayout: TensorLayout,
    BoxesEngine: TensorEngine = DefaultEngine[element_width=1],
    ScoresEngine: TensorEngine = DefaultEngine[element_width=1],
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    boxes: ImmTileTensor[
        dtype, BoxesLayout, ImmutAnyOrigin, Engine=BoxesEngine
    ],
    scores: ImmTileTensor[
        dtype, ScoresLayout, ImmutAnyOrigin, Engine=ScoresEngine
    ],
    output: TileTensor[
        mut=True, .int64, OutputLayout, MutAnyOrigin, Engine=OutputEngine
    ],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    ctx: DeviceContext,
) raises:
    """Runs NMS on `target`, dispatching to the GPU kernel when possible.

    The GPU path covers float32 inputs with ``num_boxes`` within the
    shared-memory candidate budget; anything else falls back to the CPU
    implementation on host copies of the tensors.

    Parameters:
        dtype: The data type for box coordinates and scores.
        target: Compilation target string.
        BoxesLayout: Layout type of the boxes tensor.
        ScoresLayout: Layout type of the scores tensor.
        OutputLayout: Layout type of the output tensor.
        BoxesEngine: Engine type of the boxes tensor.
        ScoresEngine: Engine type of the scores tensor.
        OutputEngine: Engine type of the output tensor.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape
            `(batch, num_boxes, 4)` where each box is `[y1, x1, y2, x2]`.
        scores: Rank-3 tensor of scores with shape
            `(batch, num_classes, num_boxes)`.
        output: Rank-2 output tensor of selected rows `[b, c, box]`,
            sized to the exact selection count by the host.
        max_output_boxes_per_class: Maximum number of boxes to select per
            class.
        iou_threshold: IoU threshold for suppression.
        score_threshold: Minimum score for a box to be considered.
        ctx: Device context used to enqueue GPU work.
    """
    comptime assert boxes.flat_rank == 3, "boxes must be rank 3"
    comptime assert scores.flat_rank == 3, "scores must be rank 3"
    comptime assert output.flat_rank == 2, "output must be rank 2"

    var num_boxes = Int(boxes.dim[1]())

    comptime if is_cpu[target]():
        # Host tensors: run the CPU kernel directly, no copies.
        non_max_suppression[dtype](
            boxes,
            scores,
            output,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )
    elif ctx.target.is_apple_gpu():
        # The kernel's shared-memory candidate state (~36 KiB at the
        # 4096-box budget) exceeds Metal's 32 KiB threadgroup limit,
        # so Apple targets always take the host fallback.
        _nms_host_fallback[dtype, BoxesLayout, ScoresLayout, OutputLayout](
            boxes,
            scores,
            output,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ctx,
            num_boxes,
        )
    elif dtype != DType.float32:
        # Non-f32 stays on the host path; copy in and back out.
        _nms_host_fallback[dtype, BoxesLayout, ScoresLayout, OutputLayout](
            boxes,
            scores,
            output,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            ctx,
            num_boxes,
        )
    else:
        # f32 on GPU: over-budget shapes fall back, the rest run the
        # GPU kernel.
        if num_boxes > _NMS_MAX_LOCAL:
            _nms_host_fallback[dtype, BoxesLayout, ScoresLayout, OutputLayout](
                boxes,
                scores,
                output,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                ctx,
                num_boxes,
            )
        else:
            non_max_suppression_gpu[
                dtype=dtype,
                BoxesLayout=BoxesLayout,
                ScoresLayout=ScoresLayout,
                OutputLayout=OutputLayout,
                BoxesEngine=BoxesEngine,
                ScoresEngine=ScoresEngine,
                OutputEngine=OutputEngine,
            ](
                boxes,
                scores,
                output,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                ctx,
            )


def _nms_host_fallback[
    dtype: DType,
    BoxesLayout: TensorLayout,
    ScoresLayout: TensorLayout,
    OutputLayout: TensorLayout,
    BoxesEngine: TensorEngine = DefaultEngine[element_width=1],
    ScoresEngine: TensorEngine = DefaultEngine[element_width=1],
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    boxes: ImmTileTensor[
        dtype, BoxesLayout, ImmutAnyOrigin, Engine=BoxesEngine
    ],
    scores: ImmTileTensor[
        dtype, ScoresLayout, ImmutAnyOrigin, Engine=ScoresEngine
    ],
    output: TileTensor[
        mut=True, .int64, OutputLayout, MutAnyOrigin, Engine=OutputEngine
    ],
    max_output_boxes_per_class: Int,
    iou_threshold: Float32,
    score_threshold: Float32,
    ctx: DeviceContext,
    num_boxes: Int,
) raises:
    """Host NMS fallback: device -> host copies, CPU kernel, copy back."""
    var batch = Int(boxes.dim[0]())
    var n_classes = Int(scores.dim[1]())
    var out_rows = Int(output.dim[0]())
    var n_boxes_total = batch * num_boxes * 4
    var n_scores_total = batch * n_classes * num_boxes
    var n_out_total = out_rows * 3

    var boxes_h = ctx.enqueue_create_host_buffer[dtype](n_boxes_total)
    var scores_h = ctx.enqueue_create_host_buffer[dtype](n_scores_total)
    var out_h = ctx.enqueue_create_host_buffer[.int64](n_out_total)

    # Wrap the device pointers as non-owning DeviceBuffers so the typed
    # buffer overloads are selected; the raw-pointer overloads assume a
    # host source and would copy the wrong direction.
    ctx.enqueue_copy(boxes_h, boxes.to_device_buffer(ctx))
    ctx.enqueue_copy(scores_h, scores.to_device_buffer(ctx))
    ctx.synchronize()

    non_max_suppression[dtype](
        TileTensor(boxes_h, boxes.layout),
        TileTensor(scores_h, scores.layout),
        TileTensor(out_h, output.layout),
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    )

    ctx.enqueue_copy(output.to_device_buffer(ctx), out_h)
