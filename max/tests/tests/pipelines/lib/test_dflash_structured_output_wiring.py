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
"""Structured-output bitmask wiring for the unified spec-decode inputs.

A spec-decode graph declares the constrained-decoding bitmask input
triple (pinned_bitmask, wait_payload, device_bitmask_scratch) when
structured output is enabled, and the ``*Inputs.buffers`` ABI tail must
pack the same triple so the buffer list lines up with the compiled graph
signature.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.lib.interfaces.pipeline_model import UnifiedSpecDecodeInputs
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
)


def _kv_params() -> MHAKVCacheParams:
    """Minimal single-device MHA KV cache leaf for input-type construction."""
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=64,
        num_layers=2,
        devices=[DeviceRef.GPU()],
        n_kv_heads=8,
    )


def _tail_sentinels() -> dict[str, object]:
    return {
        "draft_tokens": object(),
        "seed": object(),
        "temperature": object(),
        "top_k": object(),
        "max_k": object(),
        "top_p": object(),
        "min_top_p": object(),
    }


_BITMASK_SENTINELS = {
    "pinned_bitmask": object(),
    "wait_payload": object(),
    "device_bitmask_scratch": object(),
}


@pytest.mark.parametrize("include_in_thinking_phase", [False, True])
@pytest.mark.parametrize("enable_structured_output", [False, True])
def test_tail_buffers_match_input_types_lockstep(
    include_in_thinking_phase: bool,
    enable_structured_output: bool,
) -> None:
    """The ABI tail must stay in lockstep with the graph-signature tail.

    ``build_spec_decode_input_types`` declares three leading inputs
    (tokens, input_row_offsets, return_n_logits) followed by the
    flattened KV tree in the non-distributed layout; everything after
    them is the spec-decode tail that ``_spec_decode_tail_buffers``
    packs.
    """
    kv_params = _kv_params()
    types = build_spec_decode_input_types(
        SpecDecodeInputTypeSpec(
            devices=[DeviceRef.GPU()],
            distributed=False,
            include_in_thinking_phase=include_in_thinking_phase,
            enable_structured_output=enable_structured_output,
        ),
        kv_params=kv_params,
    )
    tail_types = types[3 + len(kv_params.flattened_kv_inputs()) :]

    inputs = UnifiedSpecDecodeInputs(
        in_thinking_phase=object(),  # type: ignore[arg-type]
        structured_output=enable_structured_output,
        **_tail_sentinels(),  # type: ignore[arg-type]
        **_BITMASK_SENTINELS,  # type: ignore[arg-type]
    )
    tail_buffers = inputs._spec_decode_tail_buffers(
        include_in_thinking_phase=include_in_thinking_phase,
    )
    assert len(tail_buffers) == len(tail_types)
