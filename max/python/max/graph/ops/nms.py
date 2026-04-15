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
"""Op implementation for non-maximum suppression."""

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..dim import Dim
from ..graph import Graph
from ..type import TensorType
from ..value import TensorValue, TensorValueLike


def non_maximum_suppression(
    boxes: TensorValueLike,
    scores: TensorValueLike,
    max_output_boxes_per_class: TensorValueLike,
    iou_threshold: TensorValueLike,
    score_threshold: TensorValueLike,
    out_dim: str = "num_selected",
) -> TensorValue:
    """Filters boxes with high intersection-over-union (IoU).

    Applies greedy non-maximum suppression independently per (batch, class)
    pair.  For each pair the algorithm:

    1. Discards boxes whose score is at or below ``score_threshold``.
    2. Sorts remaining boxes by score in descending order.
    3. Greedily selects boxes, suppressing any later candidate whose IoU with
       an already-selected box exceeds ``iou_threshold``.
    4. Stops after ``max_output_boxes_per_class`` selections per pair.

    Boxes use ``[y1, x1, y2, x2]`` corner format.  Coordinates may be
    normalised or absolute; the op handles both.

    Args:
        boxes: Input boxes tensor of shape ``[batch, num_boxes, 4]`` (float).
        scores: Per-class scores of shape ``[batch, num_classes, num_boxes]``
            (float, same dtype as ``boxes``).
        max_output_boxes_per_class: Scalar int64 tensor — maximum number of
            boxes to select per (batch, class) pair.
        iou_threshold: Scalar float tensor — IoU suppression threshold.
        score_threshold: Scalar float tensor — minimum score to consider.
        out_dim: Name for the dynamic output dimension (number of selected
            boxes).  Defaults to ``"num_selected"``.

    Returns:
        An int64 tensor of shape ``[out_dim, 3]`` where each row is
        ``[batch_index, class_index, box_index]``.
    """
    boxes = TensorValue(boxes)
    scores = TensorValue(scores)
    max_output_boxes_per_class = TensorValue(max_output_boxes_per_class)
    iou_threshold = TensorValue(iou_threshold)
    score_threshold = TensorValue(score_threshold)

    result_type = TensorType(
        dtype=DType.int64,
        shape=[Dim(out_dim), 3],
        device=boxes.device,
    )

    return Graph.current._add_op_generated(
        rmo.MoNonMaximumSuppressionOp,
        result_type.to_mlir(),
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
