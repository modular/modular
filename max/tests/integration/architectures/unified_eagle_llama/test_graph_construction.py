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
import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.unified_eagle_llama3.model_config import (
    UnifiedEagleLlama3Config,
)
from max.pipelines.architectures.unified_eagle_llama3.unified_eagle_llama3 import (
    UnifiedEagleLlama3,
)
from max.pipelines.lib.config import SpeculativeConfig


def create_dummy_llama3_config(layers: int) -> Llama3Config:
    return Llama3Config(
        hidden_size=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=layers,
        rope_theta=1234.0,
        rope_scaling_params=None,
        max_seq_len=2048,
        intermediate_size=256,
        interleaved_rope_weights=True,
        vocab_size=128256,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=4,
            head_dim=2,
            num_layers=layers,
            devices=[DeviceRef.GPU(0)],
            data_parallel_degree=1,
            speculative_method="eagle",
            num_draft_tokens=1,
        ),
        attention_multiplier=1.0,
        embedding_multiplier=2.0,
        residual_multiplier=3.0,
        rms_norm_eps=4.0,
        clip_qkv=6.0,
        norm_method="rms_norm",
        devices=[DeviceRef.GPU(0)],
    )


def create_dummy_eagle_llama3_config(
    enable_structured_output: bool = False,
    num_speculative_tokens: int = 1,
) -> UnifiedEagleLlama3Config:
    return UnifiedEagleLlama3Config(
        target=create_dummy_llama3_config(layers=8),
        draft=create_dummy_llama3_config(layers=1),
        speculative_config=SpeculativeConfig(
            num_speculative_tokens=num_speculative_tokens
        ),
        enable_structured_output=enable_structured_output,
    )


def build_dummy_eagle_llama3_graph(
    num_speculative_tokens: int,
) -> Graph:
    """Traces the unified graph the way ``load_model`` does.

    ``state_dict`` is what stamps each weight with its qualified name, so it
    has to run before the trace or every weight reaches the graph under its
    local name and the second one collides.
    """
    model = UnifiedEagleLlama3(
        create_dummy_eagle_llama3_config(
            num_speculative_tokens=num_speculative_tokens
        )
    )
    model.state_dict()

    with Graph(
        "unified_eagle_llama3", input_types=model.input_types()
    ) as graph:
        graph_inputs = model.decode_inputs(graph.inputs)
        graph.output(
            *model(
                tokens=graph_inputs.tokens,
                input_row_offsets=graph_inputs.input_row_offsets,
                draft_tokens=graph_inputs.draft_tokens,
                kv_collections=graph_inputs.kv("target"),
                draft_kv_collections=graph_inputs.kv("draft"),
                return_n_logits=graph_inputs.return_n_logits,
                seed=graph_inputs.seed,
                temperature=graph_inputs.temperature,
                top_k=graph_inputs.top_k,
                max_k=graph_inputs.max_k,
                top_p=graph_inputs.top_p,
                min_top_p=graph_inputs.min_top_p,
                pinned_bitmask=graph_inputs.pinned_bitmask,
                wait_payload=graph_inputs.wait_payload,
                device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
            )
        )
    return graph


def test_graph_construction() -> None:
    config = create_dummy_eagle_llama3_config()
    model = UnifiedEagleLlama3(config)

    state_dict = model.state_dict()

    # State dict must not be empty.
    assert state_dict, "State dict must not be empty"
    # Weights must be namespaced under "target." or "draft." prefixes.
    assert all(
        weight.startswith(("target.", "draft.")) for weight in state_dict
    )
    assert "draft.layers.0.mlp.up_proj.weight" in state_dict
    assert "target.layers.7.mlp.up_proj.weight" in state_dict

    # Shared weights (embed_tokens, lm_head) should appear under target.
    assert "target.embed_tokens.weight" in state_dict
    assert "target.lm_head.weight" in state_dict

    # Verify input types include draft_tokens and draft_cache_lengths.
    input_types = model.input_types()
    # Expected: tokens, input_row_offsets, return_n_logits,
    #           + target KV (8 fields) + draft KV (8 fields) + draft_tokens,
    #           + rng seed, + sampling params (temperature, top_k, max_k, top_p, min_top_p)
    assert len(input_types) == 26, (
        f"Expected 26 input types, got {len(input_types)}"
    )

    # Smoke test that graph construction (not compilation) works
    with Graph(
        "unified_eagle_llama3", input_types=model.input_types()
    ) as graph:
        graph_inputs = model.decode_inputs(graph.inputs)
        outputs = model(
            tokens=graph_inputs.tokens,
            input_row_offsets=graph_inputs.input_row_offsets,
            draft_tokens=graph_inputs.draft_tokens,
            kv_collections=graph_inputs.kv("target"),
            draft_kv_collections=graph_inputs.kv("draft"),
            return_n_logits=graph_inputs.return_n_logits,
            seed=graph_inputs.seed,
            temperature=graph_inputs.temperature,
            top_k=graph_inputs.top_k,
            max_k=graph_inputs.max_k,
            top_p=graph_inputs.top_p,
            min_top_p=graph_inputs.min_top_p,
        )
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        graph.output(*outputs)

    # Verify profile_scope labels and colors are present in the built IR.
    asm = graph._mlir_op.get_asm(enable_debug_info=True)
    for needle in (
        'profile_scope<"target_forward"',
        'profile_scope<"verify_and_sample"',
        'profile_scope<"draft_forward"',
        'profile_scope<"draft_step_0"',
        'color = "orange"',
    ):
        assert needle in asm, f"missing profile scope marker: {needle}"


def test_input_types_with_structured_output() -> None:
    """Test that input types include the bitmask triple when structured
    output is enabled.

    The overlap path binds three inputs in order: pinned bitmask source,
    the int64[2] wait payload consumed by ``mo.wait_host_value_with_dep``,
    and the device-side bitmask scratch destination.
    """
    config = create_dummy_eagle_llama3_config(enable_structured_output=True)
    model = UnifiedEagleLlama3(config)

    # Verify input types include the bitmask triple when structured
    # output is enabled.
    input_types = model.input_types()
    # Expected: 26 mandatory inputs + 3 bitmask (pinned, wait_payload,
    # device_bitmask_scratch) = 29 total
    assert len(input_types) == 29, (
        f"Expected 29 input types (with bitmask triple), got {len(input_types)}"
    )

    # The trailing three inputs are pinned_bitmask (packed int32 tensor),
    # wait_payload (int64 buffer), and device_bitmask_scratch (packed int32
    # buffer). The bitmask is stored as packed int32 (one bit per vocab token)
    # so the GPU acceptance sampler unpacks and applies it in one fused pass.
    pinned_type = input_types[-3]
    payload_type = input_types[-2]
    scratch_type = input_types[-1]
    assert pinned_type.dtype.to_numpy() == "int32", (
        f"Expected pinned bitmask dtype int32, got {pinned_type.dtype}"
    )
    assert payload_type.dtype.to_numpy() == "int64", (
        f"Expected wait_payload dtype int64, got {payload_type.dtype}"
    )
    assert scratch_type.dtype.to_numpy() == "int32", (
        f"Expected device_bitmask_scratch dtype int32, got {scratch_type.dtype}"
    )


@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3, 5])
def test_graph_traces_at_every_speculative_width(
    num_speculative_tokens: int,
) -> None:
    """The propose loop runs one iteration per token past the first, so a
    width below 3 never feeds a proposed token back in as the next step's
    input. Widths at and above 3 do, which is where a draft token carrying
    an unnamed row dim shows up as a concat error against the hidden carry.
    """
    build_dummy_eagle_llama3_graph(num_speculative_tokens)
