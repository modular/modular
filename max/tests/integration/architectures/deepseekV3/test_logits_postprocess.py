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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from max.graph import DeviceRef
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV3 import deepseekV3


@dataclass
class FakeTensor:
    rows: int
    cols: int = 16

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)


def test_last_token_logits_keep_request_batch_in_tp_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deepseekV3.ops, "gather", _fake_gather)
    monkeypatch.setattr(deepseekV3.ops, "allgather", _fake_allgather)
    monkeypatch.setattr(deepseekV3.ops, "cast", _fake_cast)
    monkeypatch.setattr(
        deepseekV3,
        "forward_sharded_layers",
        lambda norms, tensors: list(tensors),
    )
    monkeypatch.setattr(
        deepseekV3,
        "extract_hs",
        lambda **kwargs: (),
    )
    devices = [DeviceRef.GPU(id=0), DeviceRef.GPU(id=1)]

    result = deepseekV3.deepseek_logits_postprocess(
        h=cast(Any, [FakeTensor(rows=4096), FakeTensor(rows=4096)]),
        input_row_offsets=cast(
            Any,
            [
                np.array([0, 4096], dtype=np.int64),
                np.array([0, 4096], dtype=np.int64),
            ],
        ),
        all_logits_input_row_offsets=None,
        return_n_logits=cast(Any, np.array([1], dtype=np.int64)),
        norm_shards=[lambda tensor: tensor, lambda tensor: tensor],
        lm_head=cast(Any, _fake_lm_head),
        signal_buffers=[],
        devices=devices,
        is_data_parallel_attention=False,
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.NONE,
    )

    last_logits = result[0]
    assert last_logits.shape[0] == 1


def test_last_token_logits_allgather_batch_in_dp_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deepseekV3.ops, "gather", _fake_gather)
    monkeypatch.setattr(deepseekV3.ops, "allgather", _fake_allgather)
    monkeypatch.setattr(deepseekV3.ops, "cast", _fake_cast)
    monkeypatch.setattr(
        deepseekV3,
        "forward_sharded_layers",
        lambda norms, tensors: list(tensors),
    )
    monkeypatch.setattr(
        deepseekV3,
        "extract_hs",
        lambda **kwargs: (),
    )
    devices = [DeviceRef.GPU(id=0), DeviceRef.GPU(id=1)]

    result = deepseekV3.deepseek_logits_postprocess(
        h=cast(Any, [FakeTensor(rows=8), FakeTensor(rows=8)]),
        input_row_offsets=cast(
            Any,
            [
                np.array([0, 4, 8], dtype=np.int64),
                np.array([0, 4, 8], dtype=np.int64),
            ],
        ),
        all_logits_input_row_offsets=None,
        return_n_logits=cast(Any, np.array([1], dtype=np.int64)),
        norm_shards=[lambda tensor: tensor, lambda tensor: tensor],
        lm_head=cast(Any, _fake_lm_head),
        signal_buffers=[],
        devices=devices,
        is_data_parallel_attention=True,
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.NONE,
    )

    last_logits = result[0]
    assert last_logits.shape[0] == 4


def _fake_gather(source: FakeTensor, indices: Any, axis: int) -> FakeTensor:
    assert axis == 0
    return FakeTensor(rows=len(indices), cols=source.cols)


def _fake_allgather(
    tensors: list[FakeTensor], signal_buffers: Any
) -> list[FakeTensor]:
    del signal_buffers
    gathered_rows = sum(tensor.rows for tensor in tensors)
    return [
        FakeTensor(rows=gathered_rows, cols=tensors[0].cols) for _ in tensors
    ]


def _fake_cast(tensor: FakeTensor, dtype: Any) -> FakeTensor:
    del dtype
    return tensor


def _fake_lm_head(
    tensors: list[FakeTensor], signal_buffers: Any
) -> list[FakeTensor]:
    del signal_buffers
    rows = tensors[0].rows
    return [FakeTensor(rows=rows, cols=163840)]
