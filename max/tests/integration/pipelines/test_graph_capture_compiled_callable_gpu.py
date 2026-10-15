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
"""Drives ``ServeGraphCaptureRunner`` with a real ModuleV3 compiled callable.

ModuleV3 pipeline models hand the runner a ``CompiledCallable`` rather than an
engine ``Model``. This captures, verifies and replays one through the runner
on a GPU, so the unwrap to the engine model is checked against the real API.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.experimental import compilation
from max.experimental.sharding import TensorLayout
from max.experimental.tensor import Tensor
from max.nn.kv_cache import BatchCharacteristics, MHAAttnKey
from max.pipelines.lib.graph_capture import ServeGraphCaptureRunner


def _kv_params() -> MagicMock:
    kv_params = MagicMock()
    kv_params.num_draft_tokens_per_step = 1
    kv_params.graph_capture_probe_cache_lengths.return_value = [8]
    kv_params.resolve_attn_key.side_effect = (
        lambda batch_size, q, cache_length: MHAAttnKey(
            batch_size=batch_size, max_prompt_length=q, num_partitions=1
        )
    )
    return kv_params


def _inputs(values: list[float], device: Accelerator) -> Any:
    buffer = Buffer.from_numpy(np.array(values, dtype=np.float32)).to(device)
    return SimpleNamespace(buffers=(buffer,))


def test_runner_captures_and_replays_a_compiled_callable() -> None:
    device = Accelerator()

    def double(x: Tensor) -> Tensor:
        return x * 2

    compiled = compilation.compile(double)(
        TensorLayout(DType.float32, [4], device)
    )
    warmup_inputs = _inputs([0.0, 0.0, 0.0, 0.0], device)

    @contextmanager
    def _warmup(
        batch_size: int, batch_characteristics: BatchCharacteristics
    ) -> Iterator[Any]:
        yield warmup_inputs

    runner = ServeGraphCaptureRunner(
        model=compiled,
        kv_params=_kv_params(),
        warmup_model_inputs=_warmup,
        max_cache_length_upper_bound=8,
        max_batch_size=1,
    )
    runner.warmup_pre_ready()
    assert len(runner.graph_entries) == 1
    assert len(runner._host_input_names) == len(
        compiled.engine_model.input_metadata
    )

    characteristics = runner.align(
        BatchCharacteristics(
            batch_size=1, max_prompt_length=1, max_cache_valid_length=5
        )
    )
    live = _inputs([1.0, 2.0, 3.0, 4.0], device)
    outputs = runner.replay(
        model_inputs=cast(Any, live),
        batch_characteristics=characteristics,
        debug_verify_replay=True,
    )
    device.synchronize()
    np.testing.assert_allclose(outputs.logits.to_numpy(), [2.0, 4.0, 6.0, 8.0])
