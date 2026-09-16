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

"""The fused spec graph declares M-RoPE positions exactly when it should.

``vision_config`` normally carries two facts at once: an encoder must be built,
and images can appear in a request's context. The fused speculative model
clears it to avoid compiling a tower it never calls, which used to clear
``mrope_enabled`` with it -- so the positions input silently vanished from the
export and every token after an image was placed on the static rope table.
``mrope_without_encoder`` separates the two.

That failure is invisible from the graph's behavior: it compiles, runs, and
produces fluent text. Only the declared arity shows it, which is what these
tests read.
"""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.state_cache import attn_cache
from max.pipelines.architectures.unified_mtp_qwen3_5.unified_mtp_qwen3_5 import (
    UnifiedMTPQwen3_5,
)

HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
VOCAB = 128
# Temporal, height, width: must sum to half the rotary dimension.
MROPE_SECTION = [4, 2, 2]


def _config(
    *, mrope_section: list[int] | None, without_encoder: bool
) -> Qwen3_5Config:
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=HEAD_DIM,
    )
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=2,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        # Full rotary, so `mrope_section` has half of head_dim to split.
        partial_rotary_factor=1.0,
        use_subgraphs=False,
        mrope_section=mrope_section,
        mrope_without_encoder=without_encoder,
    )


def _spec_input_types(config: Qwen3_5Config) -> tuple[object, ...]:
    """The fused graph's declared inputs, with the cache tree it is built on.

    The spec graph declares its own state pools, so its tree is attention only
    -- the same shape ``UnifiedMTPQwen3_5Model`` hands it.
    """
    attn = attn_cache(config.kv_params)
    spec_kv = MultiKVCacheParams.from_params(
        {"target": attn, "draft": replace(attn, num_layers=1)}
    )
    return tuple(UnifiedMTPQwen3_5(config).input_types(spec_kv))


def test_positions_are_declared_without_a_vision_encoder() -> None:
    """The decoupling itself: no encoder, no vision config, still M-RoPE."""
    without = _spec_input_types(
        _config(mrope_section=MROPE_SECTION, without_encoder=False)
    )
    with_positions = _spec_input_types(
        _config(mrope_section=MROPE_SECTION, without_encoder=True)
    )

    assert len(with_positions) == len(without) + 1, (
        "the positions input is the only difference the flag may make"
    )

    tail = with_positions[-1]
    assert isinstance(tail, TensorType)
    assert tail.dtype == DType.int64
    # Three axes over the merged `[real, draft_1..draft_k]` window, which is
    # longer than the token count the rest of the signature is sized by.
    assert [str(d) for d in tail.shape] == ["3", "merged_total_seq_len"]

    # Everything before it keeps its slot, so one export's map is the other's
    # prefix and the engine can drive either.
    assert with_positions[:-1] == without


def test_no_mrope_section_means_no_positions() -> None:
    """The flag cannot conjure M-RoPE onto a model that has no sections."""
    assert _spec_input_types(
        _config(mrope_section=None, without_encoder=True)
    ) == _spec_input_types(_config(mrope_section=None, without_encoder=False))
