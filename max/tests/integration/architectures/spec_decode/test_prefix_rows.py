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
"""The per-row verify-prefix gather, against a numpy reference on CPU.

Row r's slot t reads merged row ``end_r - (k_r + 1) + min(t, k_r)``;
slots past ``k_r`` repeat the row's last real row. With ``k_r = K`` for
every row the gather degenerates to ``gather_variable_rows``.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.pipelines.speculative.driver import (
    gather_prefix_rows,
    gather_variable_rows,
)
from max.tensor import Tensor, compilation
from max.tensor.sharding import DeviceMesh, TensorLayout

_K = 5
_PROMPT_LENGTHS = (1, 4, 1)
_VERIFY_LENGTHS = (5, 0, 2)
_HIDDEN = 8


def _staged(
    mesh: DeviceMesh,
) -> compilation.CompiledCallable[
    [Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]
]:
    device = mesh.devices[0]

    def gather(
        h: Tensor,
        input_row_offsets: Tensor,
        verify_lengths: Tensor,
        return_n_logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return gather_prefix_rows(
            h, input_row_offsets, verify_lengths, return_n_logits, mesh
        )

    return compilation.compile(gather)(
        TensorLayout(DType.float32, ["rows", _HIDDEN], device),
        TensorLayout(DType.uint32, [len(_PROMPT_LENGTHS) + 1], device),
        TensorLayout(DType.int32, [len(_PROMPT_LENGTHS)], device),
        TensorLayout(DType.int64, [1], CPU()),
    )


def _run(
    run: compilation.CompiledCallable[
        [Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]
    ],
    device: CPU,
    h: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, logit_offsets = run(
        Tensor.from_dlpack(h).to(device),
        Tensor.from_dlpack(offsets.astype(np.uint32)).to(device),
        Tensor.from_dlpack(lengths.astype(np.int32)).to(device),
        Tensor.from_dlpack(np.array([_K + 1], dtype=np.int64)).to(CPU()),
    )
    return rows.to_numpy(), logit_offsets.to_numpy()


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    """Returns the merged window's offsets and a row-id-valued ``h``."""
    # The merged window holds each request's prompt plus its K drafts.
    offsets = np.cumsum(
        np.concatenate(([0], np.array(_PROMPT_LENGTHS, dtype=np.int64) + _K))
    )
    total_rows = int(offsets[-1])
    h = np.repeat(np.arange(total_rows, dtype=np.float32), _HIDDEN).reshape(
        total_rows, _HIDDEN
    )
    return offsets, h


def test_prefix_rows_match_the_reference() -> None:
    device = CPU()
    mesh = DeviceMesh((device,), (1,), ("tp",))
    run = _staged(mesh)

    offsets, h = _fixture()
    lengths = np.array(_VERIFY_LENGTHS, dtype=np.int64)

    rows, logit_offsets = _run(run, device, h, offsets, lengths)

    n = _K + 1
    ends = offsets[1:]
    t = np.arange(n)
    expected = (
        ends[:, None]
        - (lengths[:, None] + 1)
        + np.minimum(t[None, :], lengths[:, None])
    )
    assert rows.shape == (len(_PROMPT_LENGTHS) * n, _HIDDEN)
    np.testing.assert_array_equal(
        rows[:, 0].reshape(len(_PROMPT_LENGTHS), n), expected
    )
    np.testing.assert_array_equal(
        logit_offsets, np.arange(0, len(_PROMPT_LENGTHS) * n + 1, n)
    )


def test_uniform_prefix_rows_equal_variable_rows() -> None:
    device = CPU()
    mesh = DeviceMesh((device,), (1,), ("tp",))

    def variable(
        h: Tensor, input_row_offsets: Tensor, return_n_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        return gather_variable_rows(h, input_row_offsets, return_n_logits, mesh)

    run_variable = compilation.compile(variable)(
        TensorLayout(DType.float32, ["rows", _HIDDEN], device),
        TensorLayout(DType.uint32, [len(_PROMPT_LENGTHS) + 1], device),
        TensorLayout(DType.int64, [1], CPU()),
    )
    run_prefix = _staged(mesh)

    offsets, h = _fixture()
    lengths = np.full(len(_PROMPT_LENGTHS), _K, dtype=np.int64)

    prefix_rows, prefix_offsets = _run(run_prefix, device, h, offsets, lengths)
    variable_rows, variable_offsets = run_variable(
        Tensor.from_dlpack(h).to(device),
        Tensor.from_dlpack(offsets.astype(np.uint32)).to(device),
        Tensor.from_dlpack(np.array([_K + 1], dtype=np.int64)).to(CPU()),
    )
    np.testing.assert_array_equal(prefix_rows, variable_rows.to_numpy())
    np.testing.assert_array_equal(prefix_offsets, variable_offsets.to_numpy())
