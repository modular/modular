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
"""The per-row verify inputs a spec-decode signature declares.

``draft_scores`` and ``draft_score_coeffs`` reach the graph as inputs, so
their place in the canonical tail is fixed: after the skippable pair when
it is present, else right after the sampling inputs.
"""

from __future__ import annotations

import pytest
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.lib.interfaces.pipeline_model import UnifiedSpecDecodeInputs
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
    declared_spec_decode_inputs,
    decode_spec_decode_input_values,
    spec_decode_input_tree,
)

CPU = DeviceRef.CPU()
K = 5


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=64,
        num_layers=2,
        devices=[CPU],
        n_kv_heads=8,
    )


def _tail_names(spec: SpecDecodeInputTypeSpec) -> list[str]:
    """The spec group's input names, in the order the signature lays them."""
    named = spec_decode_input_tree(
        spec, declared_spec_decode_inputs(spec, kv_params=_kv_params())
    )
    return list(named["spec"])


def test_the_per_row_verify_inputs_follow_the_skippable_pair() -> None:
    """Both flags on: the tail ends slot ids, block offsets, scores, coeffs."""
    spec = SpecDecodeInputTypeSpec(
        devices=[CPU],
        distributed=False,
        include_skippable_draft=True,
        include_per_row_verify=True,
        num_draft_positions=K,
    )
    names = _tail_names(spec)
    assert names[-4:] == [
        "draft_slot_ids",
        "draft_block_offsets",
        "draft_scores",
        "draft_score_coeffs",
    ]


def test_the_per_row_verify_inputs_follow_the_sampling_inputs() -> None:
    """Skippable pair off: the tail ends the sampling inputs, scores, coeffs."""
    spec = SpecDecodeInputTypeSpec(
        devices=[CPU],
        distributed=False,
        include_per_row_verify=True,
        num_draft_positions=K,
    )
    names = _tail_names(spec)
    assert names == [
        "draft_tokens",
        "seed",
        "temperature",
        "top_k",
        "max_k",
        "top_p",
        "min_top_p",
        "draft_scores",
        "draft_score_coeffs",
    ]


def test_the_declared_types_carry_the_static_draft_count() -> None:
    """The scores span the batch rows and the static ``K`` draft positions;
    the calibrator coefficients span one more."""
    spec = SpecDecodeInputTypeSpec(
        devices=[CPU],
        distributed=False,
        include_per_row_verify=True,
        num_draft_positions=K,
    )
    declared = declared_spec_decode_inputs(spec, kv_params=_kv_params())
    assert declared.draft_scores == TensorType(
        DType.float32, ["batch_size", K], device=CPU
    )
    assert declared.draft_score_coeffs == TensorType(
        DType.float32, [K + 1], device=CPU
    )


def test_the_flag_off_signature_is_unchanged() -> None:
    """Without the flag the input list is the one the old spec built."""
    kv = _kv_params()
    off = build_spec_decode_input_types(
        SpecDecodeInputTypeSpec(devices=[CPU], distributed=False),
        kv_params=kv,
    )
    before = build_spec_decode_input_types(
        SpecDecodeInputTypeSpec(
            devices=[CPU],
            distributed=False,
            include_per_row_verify=False,
            num_draft_positions=K,
        ),
        kv_params=kv,
    )
    assert off == before


def test_the_flag_needs_the_static_draft_count() -> None:
    """The scores' second dim is static, so the flag refuses a missing ``K``."""
    spec = SpecDecodeInputTypeSpec(
        devices=[CPU],
        distributed=False,
        include_per_row_verify=True,
    )
    with pytest.raises(ValueError, match="num_draft_positions"):
        build_spec_decode_input_types(spec, kv_params=_kv_params())


def test_the_inputs_round_trip_through_the_decode() -> None:
    """A graph built with the flag decodes the two inputs back by name."""
    kv = _kv_params()
    spec = SpecDecodeInputTypeSpec(
        devices=[CPU],
        distributed=False,
        include_per_row_verify=True,
        num_draft_positions=K,
    )
    types = build_spec_decode_input_types(spec, kv_params=kv)
    with Graph("per_row_verify", input_types=types) as graph:
        decoded = decode_spec_decode_input_values(
            graph.inputs, spec, kv_params=kv
        )
        assert decoded.draft_scores is not None
        assert decoded.draft_scores.type == TensorType(
            DType.float32, ["batch_size", K], device=CPU
        )
        assert decoded.draft_score_coeffs is not None
        assert decoded.draft_score_coeffs.type == TensorType(
            DType.float32, [K + 1], device=CPU
        )
        graph.output()
    # The tail's keys, in the order the graph's inputs lay them.
    named = spec_decode_input_tree(
        spec, declared_spec_decode_inputs(spec, kv_params=kv)
    )
    assert list(named["spec"])[-2:] == ["draft_scores", "draft_score_coeffs"]
    assert len(tree.leaves(named)) == len(types)


def test_max_serve_refuses_a_per_row_verify_graph() -> None:
    """MAX Serve feeds no draft scores, so packing a per-row step refuses."""
    spec = SpecDecodeInputTypeSpec(
        devices=[CPU],
        distributed=False,
        include_per_row_verify=True,
        num_draft_positions=K,
    )
    inputs = UnifiedSpecDecodeInputs(spec=spec)
    unused = Buffer.zeros([1], DType.int64)
    with pytest.raises(ValueError, match="per_row_verify is not served"):
        inputs._spec_decode_buffers(
            tokens=unused, input_row_offsets=unused, return_n_logits=unused
        )
