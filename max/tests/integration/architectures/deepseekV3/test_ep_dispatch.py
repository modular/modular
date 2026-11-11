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

"""Tests for the DeepseekV3 expert-parallel dispatch decisions."""

from __future__ import annotations

from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.driver import DeviceSpec
from max.dtype import DType
from max.pipelines.architectures.deepseekV3.ep_dispatch import (
    ep_fuse_ffn_combine_send_for_pipeline,
    ep_max_rank_send_tokens_for_pipeline,
    mtp_draft_ep_dispatch_dtype,
)
from max.pipelines.lib import (
    PipelineConfig,
    PipelineRole,
    SupportedEncoding,
)

MAX_SEND_TOKENS_PER_RANK = 128
NUM_RANKS = 8


def mock_pipeline_config(
    pipeline_role: PipelineRole,
    quantization_encoding: SupportedEncoding = "float8_e4m3fn",
) -> NonCallableMock:
    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = MagicMock()
    pipeline_config.runtime = MagicMock()
    pipeline_config.model.quantization_encoding = quantization_encoding
    pipeline_config.model.kv_cache.kv_cache_format = None
    pipeline_config.model.data_parallel_degree = NUM_RANKS
    pipeline_config.model.device_specs = [
        NonCallableMock(spec=DeviceSpec) for _ in range(NUM_RANKS)
    ]

    # Pipeline config attributes
    pipeline_config.runtime.pipeline_role = pipeline_role
    pipeline_config.model.max_length = 1024 * 1024  # ~million tokens
    pipeline_config.runtime.max_batch_total_tokens = None
    pipeline_config.runtime.ep_size = NUM_RANKS
    pipeline_config.runtime.max_batch_input_tokens = MAX_SEND_TOKENS_PER_RANK
    pipeline_config.runtime.device_graph_capture = False
    pipeline_config.runtime.ep_use_allreduce = False
    pipeline_config.runtime.ep_fuse_ffn_combine_send = False
    pipeline_config.speculative = None

    return pipeline_config


def mock_huggingface_config() -> MagicMock:
    huggingface_config = MagicMock()

    # HuggingFace config attributes
    huggingface_config.num_attention_heads = 128
    huggingface_config.qk_nope_head_dim = 128
    huggingface_config.n_routed_experts = 256
    huggingface_config.moe_intermediate_size = 2048
    huggingface_config.hidden_size = 7168

    huggingface_config.num_hidden_layers = 61
    huggingface_config.first_k_dense_replace = 1
    huggingface_config.num_nextn_predict_layers = 1
    huggingface_config.vocab_size = 129280
    huggingface_config.n_shared_experts = 1
    huggingface_config.num_experts_per_tok = 8

    return huggingface_config


@pytest.mark.parametrize(
    ("pipeline_role", "encoding", "explicit", "expected"),
    [
        # Unset: on for anything that serves prefill, off for decode-only.
        # One graph serves both phases under the default role, so it gets the
        # send for decode batches too -- deliberate, not an oversight. The
        # first row is the shipping NVFP4 MoE recipes: they set neither the
        # role nor the flag, so the whole gate rides on these two defaults.
        ("prefill_and_decode", "float4_e2m1fnx2", None, True),
        ("prefill_only", "float4_e2m1fnx2", None, True),
        ("decode_only", "float4_e2m1fnx2", None, False),
        # An explicit setting wins over the role, but NOT over a config the
        # send cannot serve.
        ("decode_only", "float4_e2m1fnx2", True, True),
        ("prefill_only", "float4_e2m1fnx2", False, False),
        ("prefill_only", "float8_e4m3fn", True, False),
    ],
)
def test_ep_fuse_ffn_combine_send_follows_role_and_capability(
    pipeline_role: PipelineRole,
    encoding: SupportedEncoding,
    explicit: bool | None,
    expected: bool,
) -> None:
    """On for prefill, and only where the graph can actually elide the tensor."""
    pipeline_config = mock_pipeline_config(pipeline_role, encoding)
    pipeline_config.runtime.ep_fuse_ffn_combine_send = explicit
    assert (
        ep_fuse_ffn_combine_send_for_pipeline(
            pipeline_config, mock_huggingface_config()
        )
        is expected
    )


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        # Allreduce routes within the device; there are no peer buffers.
        ("ep_use_allreduce", True),
    ],
)
def test_ep_fuse_ffn_combine_send_off_without_peer_buffers(
    attr: str, value: bool
) -> None:
    """A runtime setting the send cannot serve turns it off, not on."""
    pipeline_config = mock_pipeline_config("prefill_only", "float4_e2m1fnx2")
    pipeline_config.runtime.ep_fuse_ffn_combine_send = True
    setattr(pipeline_config.runtime, attr, value)
    assert not ep_fuse_ffn_combine_send_for_pipeline(
        pipeline_config, mock_huggingface_config()
    )


def test_ep_fuse_ffn_combine_send_needs_an_unfused_shared_expert() -> None:
    """The send leaves only the WAIT half of the combine to run.

    The EP forward takes that split path only when a shared expert gives it
    something to overlap, so a model without one keeps the staging buffer.
    """
    pipeline_config = mock_pipeline_config("prefill_only", "float4_e2m1fnx2")
    pipeline_config.runtime.ep_fuse_ffn_combine_send = True
    huggingface_config = mock_huggingface_config()
    huggingface_config.n_shared_experts = 0
    assert not ep_fuse_ffn_combine_send_for_pipeline(
        pipeline_config, huggingface_config
    )


def test_ep_max_rank_send_tokens_divides_the_chunk_by_the_tp_degree() -> None:
    """Each rank holds its share of the chunk for the EP dispatch."""
    pipeline_config = mock_pipeline_config("prefill_and_decode")
    # DP8 over EP8 is TP1: every rank holds its whole chunk.
    assert (
        ep_max_rank_send_tokens_for_pipeline(pipeline_config)
        == MAX_SEND_TOKENS_PER_RANK
    )
    # DP2 over EP8 is TP4: the reduce-scatter hands out ceil(130 / 4).
    pipeline_config.model.data_parallel_degree = 2
    pipeline_config.runtime.max_batch_input_tokens = 130
    assert ep_max_rank_send_tokens_for_pipeline(pipeline_config) == 33
    # Allreduce keeps every token on every rank.
    pipeline_config.runtime.ep_use_allreduce = True
    assert ep_max_rank_send_tokens_for_pipeline(pipeline_config) == 130


def _mtp_config(encoding: SupportedEncoding) -> NonCallableMock:
    pipeline_config = mock_pipeline_config("prefill_and_decode", encoding)
    pipeline_config.speculative = MagicMock()
    pipeline_config.speculative.is_mtp.return_value = True
    pipeline_config.draft_model = None
    return pipeline_config


def test_mtp_draft_dispatches_in_bfloat16_beside_an_fp4_target() -> None:
    """The NextN layer's routed experts stay unquantized in FP4 checkpoints."""
    assert (
        mtp_draft_ep_dispatch_dtype(_mtp_config("float4_e2m1fnx2"))
        == DType.bfloat16
    )


def test_mtp_draft_dispatch_dtype_only_overrides_an_fp4_target() -> None:
    assert mtp_draft_ep_dispatch_dtype(_mtp_config("float8_e4m3fn")) is None
    no_spec = mock_pipeline_config("prefill_and_decode", "float4_e2m1fnx2")
    assert mtp_draft_ep_dispatch_dtype(no_spec) is None
    not_mtp = _mtp_config("float4_e2m1fnx2")
    not_mtp.speculative.is_mtp.return_value = False
    assert mtp_draft_ep_dispatch_dtype(not_mtp) is None
