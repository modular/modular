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
"""Graph-level tests for ``ops.non_maximum_suppression`` device placement."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DevicePlacementPolicy,
    DeviceRef,
    Graph,
    TensorType,
    ops,
)

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


def _nms_reference(
    boxes: np.ndarray,
    scores: np.ndarray,
    max_output: int,
    iou_threshold: float,
    score_threshold: float,
) -> np.ndarray:
    """Serial greedy NMS with the pinned order (score desc, index asc).

    Mirrors the kernel semantics exactly: strict ``>`` on both
    thresholds, corner normalization before intersection, and no
    zero-union guard (NaN IoU never suppresses).
    """
    batch, num_classes, _ = scores.shape
    rows: list[tuple[int, int, int]] = []
    for b in range(batch):
        for c in range(num_classes):
            nw = np.maximum(boxes[b, :, :2], boxes[b, :, 2:])
            se = np.minimum(boxes[b, :, :2], boxes[b, :, 2:])
            eligible = np.flatnonzero(scores[b, c] > score_threshold)
            order = sorted(eligible, key=lambda j: (-scores[b, c, j], j))
            alive = {int(j) for j in order}
            selected = 0
            for j in order:
                if j not in alive:
                    continue
                alive.discard(j)
                rows.append((b, c, int(j)))
                selected += 1
                if selected >= max_output:
                    break
                for k in list(alive):
                    # Keep every intermediate in np.float32: both
                    # kernels compute in the input dtype, and float64
                    # promotion could flip a near-threshold decision.
                    i_nw = np.minimum(nw[j], nw[k])
                    i_se = np.maximum(se[j], se[k])
                    if not (i_nw[1] < i_se[1] or i_nw[0] < i_se[0]):
                        inter = np.prod(
                            (i_se - i_nw).astype(np.float32), dtype=np.float32
                        )
                    else:
                        inter = np.float32(0.0)
                    area_j = np.prod(
                        (se[j] - nw[j]).astype(np.float32), dtype=np.float32
                    )
                    area_k = np.prod(
                        (se[k] - nw[k]).astype(np.float32), dtype=np.float32
                    )
                    union = area_j + area_k - inter
                    # abs() only here mirrors the kernel exactly,
                    # including the NaN on zero unions (NaN >
                    # threshold is False).
                    iou = np.abs(inter) / np.abs(union)
                    if iou > iou_threshold:
                        alive.discard(k)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


@pytest.mark.parametrize("num_classes", [1, 4])
def test_nms_stays_on_device(
    session: InferenceSession, num_classes: int
) -> None:
    """``ops.non_maximum_suppression`` builds under
    ``DevicePlacementPolicy.Error`` on the input's device (no implicit
    transfer) and returns the exact greedy selection.

    The three threshold scalars are declared host operands of the op, so
    they stay on CPU while boxes/scores/output live on the input device.
    """
    rng = np.random.default_rng(seed=num_classes)
    batch, num_boxes = 2, 32
    boxes_np = rng.random((batch, num_boxes, 4)).astype(np.float32)
    # Corner-normalize so boxes are [y1, x1, y2, x2] with y2 > y1.
    boxes_np[..., 2:] = (
        boxes_np[..., :2] + np.abs(boxes_np[..., 2:] - boxes_np[..., :2]) + 0.05
    )
    scores_np = rng.random((batch, num_classes, num_boxes)).astype(np.float32)

    max_output, iou_threshold, score_threshold = 10, 0.5, 0.1
    expected = _nms_reference(
        boxes_np, scores_np, max_output, iou_threshold, score_threshold
    )

    boxes_type = TensorType(DType.float32, boxes_np.shape, device_ref)
    scores_type = TensorType(DType.float32, scores_np.shape, device_ref)
    with Graph(
        "nms_device",
        input_types=[boxes_type, scores_type],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        boxes, scores = (v.tensor for v in graph.inputs)
        result = ops.non_maximum_suppression(
            boxes,
            scores,
            ops.constant(
                np.int64(max_output), DType.int64, device=DeviceRef.CPU()
            ),
            ops.constant(
                np.float32(iou_threshold), DType.float32, device=DeviceRef.CPU()
            ),
            ops.constant(
                np.float32(score_threshold),
                DType.float32,
                device=DeviceRef.CPU(),
            ),
        )
        assert result.device == device_ref
        graph.output(result)

    model = session.load(graph)
    boxes_t = Buffer.from_numpy(boxes_np).to(model.input_devices[0])
    scores_t = Buffer.from_numpy(scores_np).to(model.input_devices[1])
    output = model.execute(boxes_t, scores_t)[0]
    assert isinstance(output, Buffer)

    rows = output.to_numpy()
    # Exactly-sized output matching the reference rows in pinned order.
    assert rows.ndim == 2 and rows.shape[1] == 3
    assert rows.dtype == np.int64
    np.testing.assert_array_equal(rows, expected)
