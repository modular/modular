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

"""Minimal smoke test for single-GPU NVFP4 MoEQuantized layer."""

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.weights import WeightData
from max.nn.activation import GatedActivation
from max.nn.legacy.moe import MoEQuantized
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from torch.utils.dlpack import from_dlpack

# Wide enough for the fused MegaFFN: its packed K1 and K2 must be multiples
# of the 128-byte k tile.
HIDDEN_DIM = 512
MOE_DIM = 256
NUM_EXPERTS = 4
NUM_EXPERTS_PER_TOKEN = 2
SEQ_LEN = 16


def _add_fp4_proj(
    weights: dict[str, torch.Tensor],
    prefix: str,
    out_dim: int,
    in_dim: int,
) -> None:
    weight = torch.randint(0, 256, (out_dim, in_dim // 2), dtype=torch.uint8)
    weight_scale = (
        torch.rand(out_dim, weight.shape[1] // 8, dtype=torch.float32) * 100
    ).to(torch.float8_e4m3fn)
    weights[f"{prefix}.weight"] = weight
    weights[f"{prefix}.weight_scale"] = weight_scale
    weights[f"{prefix}.weight_scale_2"] = (
        torch.rand((), dtype=torch.float32) * 1e-4
    )
    weights[f"{prefix}.input_scale"] = (
        torch.rand((), dtype=torch.float32) * 1e-3
    )


def _nvfp4_config(can_use_fused_swiglu: bool = False) -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
        can_use_fused_swiglu=can_use_fused_swiglu,
    )


def _random_weights() -> dict[str, WeightData | torch.Tensor]:
    raw_weights: dict[str, torch.Tensor] = {}
    raw_weights["gate.gate_score.weight"] = torch.randn(
        NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16
    )
    for i in range(NUM_EXPERTS):
        _add_fp4_proj(
            raw_weights, f"experts.{i}.gate_proj", MOE_DIM, HIDDEN_DIM
        )
        _add_fp4_proj(raw_weights, f"experts.{i}.up_proj", MOE_DIM, HIDDEN_DIM)
        _add_fp4_proj(
            raw_weights, f"experts.{i}.down_proj", HIDDEN_DIM, MOE_DIM
        )

    # Wrap float8_e4m3fn tensors as WeightData
    wrapped: dict[str, WeightData | torch.Tensor] = {}
    for key, value in raw_weights.items():
        if value.dtype == torch.float8_e4m3fn:
            wrapped[key] = WeightData(
                Buffer.from_dlpack(value.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(value.shape),
            )
        else:
            wrapped[key] = value
    return wrapped


def _build_moe_graph(
    weights: dict[str, WeightData | torch.Tensor],
    *,
    fused_swiglu_layout_without_ep: bool,
    dynamic_tokens: bool = False,
    max_routed_rows: int = 0,
    megaffn_prefill_dynamic_claim: bool = False,
    activation: GatedActivation | None = None,
) -> tuple[Graph, MoEQuantized]:
    moe = MoEQuantized(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=MOE_DIM,
        dtype=DType.uint8,
        quant_config=_nvfp4_config(
            can_use_fused_swiglu=fused_swiglu_layout_without_ep
        ),
        fused_swiglu_layout_without_ep=fused_swiglu_layout_without_ep,
        max_routed_rows=max_routed_rows,
        megaffn_prefill_dynamic_claim=megaffn_prefill_dynamic_claim,
        activation=activation if activation is not None else GatedActivation(),
    )
    moe.load_state_dict(weights)

    input_type = TensorType(
        DType.bfloat16,
        ["seq_len" if dynamic_tokens else SEQ_LEN, HIDDEN_DIM],
        device=DeviceRef.GPU(),
    )

    with Graph("MoEQuantized_NVFP4_test", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = moe(x.tensor)
        graph.output(output)
    return graph, moe


def _compile_moe(
    weights: dict[str, WeightData | torch.Tensor],
    *,
    fused_swiglu_layout_without_ep: bool,
    dynamic_tokens: bool = False,
    max_routed_rows: int = 0,
    megaffn_prefill_dynamic_claim: bool = False,
    activation: GatedActivation | None = None,
) -> Model:
    graph, moe = _build_moe_graph(
        weights,
        fused_swiglu_layout_without_ep=fused_swiglu_layout_without_ep,
        dynamic_tokens=dynamic_tokens,
        max_routed_rows=max_routed_rows,
        megaffn_prefill_dynamic_claim=megaffn_prefill_dynamic_claim,
        activation=activation,
    )
    session = InferenceSession(devices=[Accelerator()])
    return session.load(graph, weights_registry=moe.state_dict())


def _execute(model: Model, hidden_states: torch.Tensor) -> torch.Tensor:
    result = model.execute(Buffer.from_dlpack(hidden_states).to(Accelerator()))
    return from_dlpack(result[0]).clone()


def _run_moe(
    weights: dict[str, WeightData | torch.Tensor],
    hidden_states: torch.Tensor,
    *,
    fused_swiglu_layout_without_ep: bool,
) -> torch.Tensor:
    model = _compile_moe(
        weights,
        fused_swiglu_layout_without_ep=fused_swiglu_layout_without_ep,
    )
    return _execute(model, hidden_states)


@pytest.fixture(scope="module")
def swigluoai_models() -> tuple[Model, Model, Model]:
    torch.manual_seed(17)
    weights = _random_weights()
    activation = GatedActivation(
        fused_activation="swigluoai", alpha=1.702, limit=7.0
    )
    chain = _compile_moe(
        weights,
        fused_swiglu_layout_without_ep=False,
        dynamic_tokens=True,
        activation=activation,
    )
    pair = _compile_moe(
        weights,
        fused_swiglu_layout_without_ep=True,
        dynamic_tokens=True,
        activation=activation,
    )
    megaffn = _compile_moe(
        weights,
        fused_swiglu_layout_without_ep=True,
        dynamic_tokens=True,
        max_routed_rows=512 * NUM_EXPERTS_PER_TOKEN,
        activation=activation,
    )
    return chain, pair, megaffn


@pytest.mark.parametrize("fused_index", [1, 2], ids=["pair", "megaffn"])
def test_moe_nvfp4_swigluoai_bf16_matches_chain(
    swigluoai_models: tuple[Model, Model, Model], fused_index: int
) -> None:
    """Checks BF16-rounded SwiGLU-OAI parity at decode and prefill sizes."""
    torch.manual_seed(19)
    chain = swigluoai_models[0]
    fused = swigluoai_models[fused_index]
    # Top-2 routing over four experts reaches decode and both prefill tiles.
    for seq_len in (1, 100, 300):
        hidden_states = torch.randn(
            seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
        )
        expected = _execute(chain, hidden_states)
        actual = _execute(fused, hidden_states)
        assert torch.all(torch.isfinite(expected))
        assert torch.count_nonzero(expected) > 0
        assert torch.equal(actual, expected), seq_len


def test_moe_nvfp4_without_row_bound_accepts_long_prefill(
    swigluoai_models: tuple[Model, Model, Model],
) -> None:
    """Accepts an unchunked prefill beyond the bounded graph's token limit."""
    torch.manual_seed(23)
    hidden_states = torch.randn(
        1024, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    expected = _execute(swigluoai_models[0], hidden_states)
    actual = _execute(swigluoai_models[1], hidden_states)
    assert torch.all(torch.isfinite(expected))
    assert torch.equal(actual, expected)
    # Only MegaFFN enforces the compiled bound; this proves its fusion fired.
    with pytest.raises(Exception, match="max_total_m"):
        _execute(swigluoai_models[2], hidden_states)


def test_moe_nvfp4_single_gpu() -> None:
    """Verify single-GPU NVFP4 MoEQuantized builds and produces finite output."""
    torch.manual_seed(42)
    hidden_states = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    output_tensor = _run_moe(
        _random_weights(),
        hidden_states,
        fused_swiglu_layout_without_ep=False,
    )

    assert output_tensor.shape == (SEQ_LEN, HIDDEN_DIM)
    assert torch.all(torch.isfinite(output_tensor))


def test_moe_nvfp4_fused_swiglu_layout_matches_chain() -> None:
    """The opt-in fused SwiGLU layout agrees with the unfused TP chain.

    Both layers load the same checkpoint tensors; the opt-in one permutes
    gate/up in the graph and runs the fused grouped matmul + SwiGLU + quant
    kernel, so only rounding in the standalone quantize may differ.
    """
    torch.manual_seed(7)
    weights = _random_weights()
    hidden_states = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    chain = _run_moe(
        weights, hidden_states, fused_swiglu_layout_without_ep=False
    )
    fused = _run_moe(
        weights, hidden_states, fused_swiglu_layout_without_ep=True
    )

    assert torch.all(torch.isfinite(fused))
    cosine = torch.nn.functional.cosine_similarity(
        chain.float(), fused.float(), dim=-1
    )
    assert cosine.min() > 0.999, cosine


def test_moe_nvfp4_megaffn_dynamic_tokens() -> None:
    """The fused layout takes the MegaFFN at a dynamic row count.

    One compiled graph with a symbolic token count runs several plan sizes,
    including ones that leave experts empty. At the static graph's size it
    matches that graph bit for bit, and at every size it agrees with the
    unfused TP chain up to rounding. A row count past ``max_routed_rows`` is
    refused instead of overrunning the arrival counters, which also shows the
    MegaFFN fusion fired: the unfused pair has no such bound.
    """
    torch.manual_seed(11)
    weights = _random_weights()
    max_tokens = 64
    chain = _compile_moe(weights, fused_swiglu_layout_without_ep=False)
    static = _compile_moe(weights, fused_swiglu_layout_without_ep=True)
    dynamic = _compile_moe(
        weights,
        fused_swiglu_layout_without_ep=True,
        dynamic_tokens=True,
        max_routed_rows=max_tokens * NUM_EXPERTS_PER_TOKEN,
    )

    # 1 token leaves most of the 4 experts empty; 64 is the compiled bound.
    for seq_len in (1, 3, SEQ_LEN, max_tokens):
        hidden_states = torch.randn(
            seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
        )
        actual = _execute(dynamic, hidden_states)
        assert actual.shape == (seq_len, HIDDEN_DIM)
        assert torch.all(torch.isfinite(actual))

        # The static graphs only take SEQ_LEN rows: pad, then compare the
        # leading rows. Routing and quantization are per token, so padding
        # rows do not change them.
        rows = min(seq_len, SEQ_LEN)
        padded = torch.zeros(
            SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
        )
        padded[:rows] = hidden_states[:rows]
        expected_fused = _execute(static, padded)[:rows]
        expected_chain = _execute(chain, padded)[:rows]
        assert torch.equal(actual[:rows], expected_fused), seq_len
        cosine = torch.nn.functional.cosine_similarity(
            expected_chain.float(), actual[:rows].float(), dim=-1
        )
        assert cosine.min() > 0.99, (seq_len, cosine)

    too_many = torch.randn(
        max_tokens + 1, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(Exception, match="max_total_m"):
        _execute(dynamic, too_many)


def test_moe_nvfp4_megaffn_prefill_dynamic_claim() -> None:
    """The opt-in claimed prefill walk matches the round-robin walk.

    Only the layer that sets ``megaffn_prefill_dynamic_claim`` carries the
    attribute into its graph. Both walks run the same tiles, so the output is
    bit-identical at a decode size (where the flag is ignored), at the
    128-token 2-CTA prefill geometry, and at the 256-token one.
    """
    torch.manual_seed(13)
    weights = _random_weights()
    max_tokens = 512
    bound = max_tokens * NUM_EXPERTS_PER_TOKEN
    claimed_graph, _ = _build_moe_graph(
        weights,
        fused_swiglu_layout_without_ep=True,
        dynamic_tokens=True,
        max_routed_rows=bound,
        megaffn_prefill_dynamic_claim=True,
    )
    unclaimed_graph, _ = _build_moe_graph(
        weights,
        fused_swiglu_layout_without_ep=True,
        dynamic_tokens=True,
        max_routed_rows=bound,
    )
    assert "prefill_dynamic_claim = true" in str(claimed_graph)
    assert "prefill_dynamic_claim" not in str(unclaimed_graph)

    claimed = _compile_moe(
        weights,
        fused_swiglu_layout_without_ep=True,
        dynamic_tokens=True,
        max_routed_rows=bound,
        megaffn_prefill_dynamic_claim=True,
    )
    unclaimed = _compile_moe(
        weights,
        fused_swiglu_layout_without_ep=True,
        dynamic_tokens=True,
        max_routed_rows=bound,
    )
    # Rows per expert are about tokens / 2 with 4 experts and top-2 routing:
    # decode up to 32, 128-token prefill tiles up to 128, 256-token above.
    for seq_len in (8, 100, 300, max_tokens):
        hidden_states = torch.randn(
            seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
        )
        expected = _execute(unclaimed, hidden_states)
        # Repeat so consecutive launches share one arrival buffer.
        for _ in range(3):
            actual = _execute(claimed, hidden_states)
            assert torch.equal(actual, expected), seq_len
