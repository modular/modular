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
"""ModuleV3 embedding inputs match the types their graph declares."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from max.driver import CPU
from max.graph import DeviceRef
from max.pipelines.architectures.qwen3_embedding_modulev3.batch_processor import (
    Qwen3EmbeddingModuleV3BatchProcessor,
)


def test_embedding_inputs_match_their_symbolic_types() -> None:
    """Each built input has the dtype and device ``get_symbolic_inputs`` declares.

    ``compile()`` binds the graph to the symbolic types, and execution
    rejects an input whose dtype or device differs from them. The declared
    types target a GPU so a CPU-only host still distinguishes the token
    device from the host-side row offsets and ``return_n_logits``.
    """
    processor = object.__new__(Qwen3EmbeddingModuleV3BatchProcessor)
    processor.runtime = MagicMock(devices=[CPU()])
    contexts = [
        MagicMock(tokens=MagicMock(active=np.array([1, 2, 3]))),
        MagicMock(tokens=MagicMock(active=np.array([4, 5]))),
    ]

    inputs = processor.prepare_initial_token_inputs([contexts])
    built = [inputs.tokens, inputs.input_row_offsets, inputs.return_n_logits]
    device_ref = DeviceRef.GPU()
    tokens, row_offsets, return_n_logits = processor.get_symbolic_inputs(
        kv_params=MagicMock(),
        device_refs=[device_ref],
    )

    assert tokens.device == device_ref
    assert row_offsets.device == DeviceRef.CPU()
    assert return_n_logits.device == DeviceRef.CPU()
    assert DeviceRef.from_device(inputs.input_row_offsets.device).is_cpu()
    assert DeviceRef.from_device(inputs.return_n_logits.device).is_cpu()
    for buffer, symbolic in zip(
        built, [tokens, row_offsets, return_n_logits], strict=True
    ):
        assert buffer.dtype == symbolic.dtype
