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

"""Pins the vision position embedding across a clip's temporal slices.

``BilinearInterpolationPositionEmbedding`` interpolates one position grid per
FRAME -- ``sum(h * w)`` rows -- and ``mo.spatial_merge`` replicates it across
each grid item's ``t`` temporal patches, emitting ``sum(t * h * w)``. The two
lengths coincide only for still images, so nothing in the image path can see a
mistake here.

Two shapes carry the numerics:

- a clip's rows must repeat its single interpolated frame ``t`` times, and
- a grid item FOLLOWING a clip must still read its own position rows. The
  kernel advances its input cursor by ``h * w`` and its output cursor by
  ``t * h_out * w_out``; declaring the output as "one row per input row" makes
  a single clip come out right while corrupting whatever follows it, which is
  the failure a one-item batch cannot show.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_bilinear_interpolation_weights_and_indices,
)
from max.pipelines.architectures.qwen3vl_moe.nn.visual_transformer import (
    BilinearInterpolationPositionEmbedding,
)

HIDDEN_SIZE = 8
MERGE_SIZE = 2
NUM_GRID_PER_SIDE = 8
NUM_POSITION_EMBEDDINGS = NUM_GRID_PER_SIDE**2


def _reference(
    table: npt.NDArray[np.float32],
    indices: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float32],
    grid_thw: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """The layer's semantics in numpy, per the reference's torch formulation.

    Sums the four bilinear neighbours, then for each grid item reshapes its
    ``(h, w)`` plane into merge blocks -- ``permute(0, 1, 3, 2, 4, 5)`` on
    ``(t, h/m, m, w/m, m, hidden)`` -- and repeats the result ``t`` times.
    """
    weighted = (table[indices] * weights).sum(axis=0)

    out: list[npt.NDArray[np.float32]] = []
    offset = 0
    for t, h, w in grid_thw:
        plane = weighted[offset : offset + h * w].reshape(h, w, HIDDEN_SIZE)
        offset += h * w
        merged = (
            plane.reshape(
                h // MERGE_SIZE,
                MERGE_SIZE,
                w // MERGE_SIZE,
                MERGE_SIZE,
                HIDDEN_SIZE,
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(h * w, HIDDEN_SIZE)
        )
        out.append(np.tile(merged, (t, 1)))
    return np.concatenate(out).astype(np.float32)


def _run_on_gpu(
    grid_thw: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Executes the layer for ``grid_thw`` and returns ``(actual, expected)``."""
    raw_indices, raw_weights = get_bilinear_interpolation_weights_and_indices(
        grid_thw=grid_thw, num_grid_per_side=NUM_GRID_PER_SIDE
    )
    # As `vision_packing` does: the helper's arithmetic promotes to float64.
    indices = raw_indices.astype(np.int64)
    weights = raw_weights.astype(np.float32)
    pos_rows = int(sum(h * w for _, h, w in grid_thw))
    patch_rows = int(sum(t * h * w for t, h, w in grid_thw))
    assert indices.shape == (4, pos_rows)

    rng = np.random.default_rng(0)
    table = rng.standard_normal(
        (NUM_POSITION_EMBEDDINGS, HIDDEN_SIZE), dtype=np.float32
    )

    device = DeviceRef.GPU(0)
    layer = BilinearInterpolationPositionEmbedding(
        dtype=DType.float32,
        device=device,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        hidden_size=HIDDEN_SIZE,
        spatial_merge_size=MERGE_SIZE,
    )

    with Graph(
        "bilinear_pos_embed",
        input_types=[
            TensorType(DType.int64, shape=[4, "pos_len"], device=device),
            TensorType(DType.float32, shape=[4, "pos_len", 1], device=device),
            TensorType(DType.int64, shape=["n_grids", 3], device=device),
        ],
    ) as graph:
        idxs_in, weights_in, grid_in = (inp.tensor for inp in graph.inputs)
        graph.output(
            layer(
                idxs=idxs_in,
                weights=weights_in,
                grid_thw=grid_in,
                out_rows=patch_rows,
            )
        )

    accelerator = Accelerator(0)
    session = InferenceSession(devices=[accelerator])
    # The layer is the graph root here, so the embedding's weight keeps its
    # own bare name rather than a module-path prefix.
    model = session.load(graph, weights_registry={"weight": table})
    output = model.execute(
        Buffer.from_numpy(indices).to(accelerator),
        Buffer.from_numpy(weights).to(accelerator),
        Buffer.from_numpy(grid_thw.astype(np.int64)).to(accelerator),
    )[0]
    assert isinstance(output, Buffer)
    actual = output.to(CPU()).to_numpy().astype(np.float32)
    return actual, _reference(table, indices, weights, grid_thw)


@pytest.mark.parametrize(
    "grid_thw",
    [
        pytest.param([(1, 4, 4)], id="one_image"),
        pytest.param([(1, 4, 4), (1, 2, 2)], id="two_images"),
    ],
)
def test_still_image_layout(grid_thw: list[tuple[int, int, int]]) -> None:
    """A ``t == 1`` batch keeps the reference merge-block layout.

    Compared with a tolerance rather than exactly: the graph and the numpy
    reference sum the four bilinear neighbours in different float32
    accumulation orders. A layout mistake moves whole rows, which is orders of
    magnitude above this bound.
    """
    actual, expected = _run_on_gpu(np.array(grid_thw, dtype=np.int64))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_clip_replicates_position_grid_across_temporal_patches() -> None:
    """A clip's ``t`` temporal patches each carry the same position rows."""
    actual, expected = _run_on_gpu(np.array([(3, 4, 4)], dtype=np.int64))
    assert actual.shape == (3 * 4 * 4, HIDDEN_SIZE)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    # The three slices are copies of one another, which is what makes the
    # position array legitimately shorter than pixel_values.
    slices = actual.reshape(3, 4 * 4, HIDDEN_SIZE)
    np.testing.assert_array_equal(slices[0], slices[1])
    np.testing.assert_array_equal(slices[0], slices[2])


def test_grid_after_a_clip_reads_its_own_position_rows() -> None:
    """The item following a clip is not fed the clip's position rows.

    The regression an "output rows == input rows" declaration hides: the
    kernel's input cursor is already correct, so only a batch with a grid
    AFTER a ``t > 1`` grid can observe the wrong ``offset_in``.
    """
    grid_thw = np.array([(2, 4, 4), (1, 2, 2)], dtype=np.int64)
    actual, expected = _run_on_gpu(grid_thw)
    assert actual.shape == (2 * 4 * 4 + 1 * 2 * 2, HIDDEN_SIZE)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    # The trailing image's rows must differ from the clip's, otherwise the
    # kernel re-read the clip's position plane.
    trailing = actual[2 * 4 * 4 :]
    assert not np.array_equal(trailing, actual[: 1 * 2 * 2])
