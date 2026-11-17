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
"""The drafter's numerics against the torch reference and TorchSpec's golden.

One random-weight tiny drafter runs twice: through the real ModuleV3 module
on the GPU (the fused MLA store and prefill/decode attention the fused
spec-decode graph binds, over a compressed latent cache) and through an
fp32 CPU oracle (``torch_reference.K3DSparkReference``, the published
TorchSpec drafter as an explicit forward). The comparison isolates
compute: both sides load the same bf16-rounded weights and read the same
bf16-rounded taps, so a divergence is a wiring or kernel difference, not
quantization noise. The module runs at TP1 and TP2, each at both latent
cache dtypes the drafter serves: bf16, which routes the block call to the
one-shot decompression prefill arm at every head count, and fp8
(production), which folds at this tiny head count -- every fused arm the
pairing can land on. The fp32 reference is a stronger TP oracle than a
TP1-vs-TP2 self-comparison, which any shared sharding bug passes.

The chain the module drew must also disagree with the unbiased argmax of
its own logits somewhere, or the Markov bias this instance carries never
flipped a pick and the draft-equality assert is not load-bearing.

RED controls sabotage the module the same way and must each fail the
gate the clean runs pass, or the oracle could not catch the defect class
they stand for: the context store dropped (a store that silently stops
storing), the block mask causal instead of null (the block-diffusion mask
mis-wired), and the context store's layernorm gain dropped (gamma
mis-wiring or double normalisation -- invisible while every gamma was
ones, so the fixture now draws non-trivial gains).

The golden tests replay outputs recorded from TorchSpec's own
``K3DSparkModel`` (see the generator in the test tree's ``tools``
package) at a near and a far
(60000) anchor position: the reference must reproduce the recorded
logits to 1e-4 and the greedy block bit for bit, which pins the YaRN table
and the interleaved rope far past what the near position exercises. The
golden records the outputs, the inputs and the weight seed, not the
weights themselves (the two ``kv_b`` tensors alone are 2 MB, over the
repo's testdata size limit): the replay regenerates them bit for bit
from the seed with ``make_tiny_drafter.draw_state_dict``, the same draw
that produced the weights the reference consumed when the outputs were
recorded.
"""

from __future__ import annotations

import os

os.environ.setdefault("MODULAR_DEBUG", "source-tracebacks")

import json
import pathlib

import numpy as np
import pytest
import torch
from _dspark_test_data import TINY_DRAFT_CONFIG
from _dspark_test_utils import (
    allocate_dspark_cache,
    set_dspark_weight,
)
from _tp_model_test_utils import rel_error
from make_tiny_drafter import draw_state_dict
from max.driver import CPU, Accelerator, accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.mesh_axis import TP
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.pipelines.architectures.kimi_k3_modulev3.dspark import (
    dspark as dspark_module,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    K3DSparkDraftConfig,
    parse_k3_dspark_config,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.dspark import (
    K3DSpark,
    tensor_parallel_k3_dspark,
)
from max.pipelines.architectures.kimi_k3_modulev3.spec_mla import (
    kimi_k3_mla_buffer_size,
)
from max.support.math import ceildiv
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh, NamedMapping
from safetensors.torch import load_file
from torch_reference import K3DSparkReference, expected_state_dict_shapes

_NORM_NAMES = ("context_norm.weight", "final_norm.weight")
_EMBEDDING_NAMES = ("embed_tokens.weight", "markov_head.markov_w1.weight")
# Big enough that the bigram bias moves argmaxes, so the chain's sequential
# dependency is exercised, without drowning the backbone's logits.
_MARKOV_W2_SCALE = 12.0

# Both latent cache dtypes the drafter serves, one per fused arm they
# route the block call to on AMD at this head count: bf16 takes the
# one-shot decompression prefill, fp8 (production) the decode fold.
_CACHE_DTYPES = [
    pytest.param(DType.bfloat16, id="bf16_prefill"),
    pytest.param(DType.float8_e4m3fn, id="fp8_fold"),
]
# The production page: the AMD MLA kernels stage the KV in 128-wide tiles,
# which the page size must divide (the fused kernels refuse a smaller page
# at compile time), and the pairing runs the drafter on the target's own
# page size.
_PAGE_SIZE = 128

GOLDEN_DIR = pathlib.Path(__file__).parent / "testdata"
# Both fp32 forwards differ only in reduction order (einsum/SDPA); the
# measured drift is 2.4e-6 on the recorded block. 1e-4 leaves ~40x headroom
# while sitting orders of magnitude below any semantic error -- a wrong rope
# layout or softmax scale moves logits by >= 0.02 (measured by sabotage).
GOLDEN_LOGITS_ATOL = 1e-4


def _round_to_bf16(x: np.ndarray) -> np.ndarray:
    """Rounds fp32 to bf16, nearest-even, the way the module's weights sit."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _draw(
    name: str, shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """One random tensor, in the reference generator's own distributions."""
    if name.endswith("layernorm.weight") or name in _NORM_NAMES:
        # Gains drawn around 1, at bf16 like every weight: with all-ones
        # gammas a dropped or swapped layernorm gain is numerically
        # invisible (the gain is a no-op), so gamma mis-wiring and
        # double normalisation -- defects this lane's in-cache gamma
        # wiring exists to catch -- could not move the oracle.
        return _round_to_bf16(rng.uniform(0.5, 1.5, shape).astype(np.float32))
    if name in _EMBEDDING_NAMES:
        return _round_to_bf16(rng.standard_normal(shape).astype(np.float32))
    bound = 1.0 / np.sqrt(shape[-1])
    drawn = rng.uniform(-bound, bound, shape).astype(np.float32)
    if name == "markov_head.markov_w2.weight":
        drawn = drawn * _MARKOV_W2_SCALE
    return _round_to_bf16(drawn)


def _weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Every tensor the reference validates, module and confidence heads."""
    shapes = expected_state_dict_shapes(TINY_DRAFT_CONFIG)
    return {
        name: _draw(name, tuple(shape), rng)
        for name, shape in sorted(shapes.items())
    }


def _run_module(
    config: K3DSparkDraftConfig,
    weights: dict[str, np.ndarray],
    head: np.ndarray,
    taps: np.ndarray,
    anchor_token: int,
    anchor_position: int,
    num_devices: int,
    cache_dtype: DType,
    *,
    store_context: bool = True,
    store_gamma_ones: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One draft step through the real module on a ``num_devices`` mesh.

    The context rows sit at positions ``0..anchor_position - 1`` and the
    block rows at ``anchor_position..+block_size - 1``, matching the cache
    layout the block driver runs: the materialize store writes the context
    at the pre-iteration length, and the block forward runs at the
    committed length, roping each row at its own absolute position.

    ``store_gamma_ones`` is the RED control's gamma mutant: the context
    store applies no layernorm gain, the module's own drawn gamma restored
    for the block call that follows, so only the stored context rows are
    wrong.
    """
    block = config.block_size
    devices = [Accelerator(i) for i in range(num_devices)]
    mesh = DeviceMesh(tuple(devices), (num_devices,), ("tp",))
    kv_params = MLAKVCacheParams(
        dtype=cache_dtype,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.from_device(device) for device in devices],
        page_size=_PAGE_SIZE,
        num_q_heads=config.num_attention_heads,
    )
    # The plan cap production sizing gives this drafter: the block call's
    # rows are the block's width, and the arm the cache dtype picks sizes
    # what the cap must cover.
    window = ceildiv(anchor_position + block, _PAGE_SIZE) * _PAGE_SIZE
    mla_buffer_size = kimi_k3_mla_buffer_size(
        max_batch_context_length=window,
        max_seq_len=anchor_position + block,
        page_size=_PAGE_SIZE,
        num_draft_tokens=block - 1,
        max_batch_size=1,
        num_q_heads_per_device=config.num_attention_heads // num_devices,
        cache_dtype=cache_dtype,
        device_api=accelerator_api(),
    )
    taps_bf16 = _round_to_bf16(taps)
    with default_device(mesh), default_dtype(DType.bfloat16):
        draft = K3DSpark(
            config,
            kv_params,
            max_seq_len=anchor_position + block + 1,
            mla_buffer_size=mla_buffer_size,
        )
        for name, value in weights.items():
            if name.startswith("confidence_head."):
                continue
            set_dspark_weight(draft, name, Tensor(value, dtype=DType.float32))
        if num_devices > 1:
            tensor_parallel_k3_dspark(draft, mesh)
        cache = allocate_dspark_cache(kv_params, [anchor_position + block])
        offsets = Tensor(
            np.array([0, anchor_position], np.uint32), dtype=DType.uint32
        ).to(mesh)
        if store_context:
            saved_gammas: list[Tensor] = []
            if store_gamma_ones:
                saved_gammas = [
                    layer.self_attn.kv_a_layernorm.weight
                    for layer in draft.layers
                ]
                for layer in draft.layers:
                    layer.self_attn.kv_a_layernorm.weight = Tensor(
                        np.ones(config.kv_lora_rank, np.float32),
                        dtype=DType.float32,
                    ).cast(layer.self_attn.kv_a_layernorm.weight.dtype)
            draft.materialize(
                draft.project_context(
                    Tensor(taps_bf16, dtype=DType.float32)
                    .cast(DType.bfloat16)
                    .to(mesh)
                ),
                cache([anchor_position], [0]),
                offsets,
            )
            if store_gamma_ones:
                for layer, gamma in zip(
                    draft.layers, saved_gammas, strict=True
                ):
                    layer.self_attn.kv_a_layernorm.weight = gamma
        # The RED control with ``store_context`` False skips the store
        # entirely: the cache keeps its zero fill, so the block attends
        # over empty context rows.
        block_ids = np.array(
            [anchor_token] + [config.mask_token_id] * (block - 1),
            dtype=np.int64,
        )
        embeds = draft.embed_tokens(
            Tensor(block_ids, dtype=DType.int64).to(mesh)
        )
        hidden = draft.forward_block(
            embeds,
            cache([block], [anchor_position]),
            Tensor(np.array([0, block], np.uint32), dtype=DType.uint32).to(
                mesh
            ),
            # The plan extent that production threads host-side: the
            # block's rows over the cached context, page-aligned. Oversized
            # is safe (the copy zero-fills the slack); undersized silently
            # corrupts. The kernel reads it as a host scalar, so the tensor
            # must not ride the caller's default device.
            Tensor(
                np.array([window], np.int32), dtype=DType.int32, device=CPU()
            ),
        )
        head_weight = Tensor(head, dtype=DType.float32).cast(DType.bfloat16)
        if num_devices > 1:
            # The target's LM head arrives vocabulary-sharded under TP.
            head_weight = head_weight.to(NamedMapping(mesh, (TP, None)))
        logits = draft.logits(hidden, head_weight)
        # The chain runs where the head left the logits: the first device.
        drafts, _ = draft.draft_chain(
            logits,
            Tensor(np.array([anchor_token]), dtype=DType.int64).to(devices[0]),
            None,
        )
    return logits.to_numpy(), drafts.to_numpy()


def _reference_logits(
    reference: K3DSparkReference,
    taps: np.ndarray,
    anchor_token: int,
    anchor_position: int,
) -> np.ndarray:
    """The fp32 oracle's logits for the same taps the module read."""
    return reference.draft_logits(
        torch.from_numpy(_round_to_bf16(taps)),
        torch.arange(anchor_position),
        anchor_token,
        anchor_position,
    ).numpy()


def _tiny_instance(
    anchor_position: int, *, seed: int = 20261113
) -> tuple[
    K3DSparkDraftConfig,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    int,
]:
    """One random-weight tiny drafter: config, weights, head, taps, anchor."""
    config = parse_k3_dspark_config(TINY_DRAFT_CONFIG)
    rng = np.random.default_rng(seed)
    weights = _weights(rng)
    head = _round_to_bf16(
        rng.standard_normal([config.vocab_size, config.hidden_size]).astype(
            np.float32
        )
        * 0.05
    )
    taps = (
        rng.standard_normal(
            [
                anchor_position,
                config.num_context_features * config.hidden_size,
            ]
        ).astype(np.float32)
        * 0.05
    )
    anchor_token = int(rng.integers(0, config.vocab_size))
    return config, weights, head, taps, anchor_token


# The logits gate per latent cache dtype, against the fp32 reference.
# Measured on this fixture (TP1; the TP2 clean runs sit under the same
# bounds), each gate must sit between its arm's clean maximum and its
# weakest RED, both of which the suite below pins:
#
#   latent arm     clean max      RED dropped-store   RED causal   RED ones-gamma
#   bf16 prefill   6.9e-3         6.5e-2             --           2.6e-2
#   fp8 fold       1.01e-2        6.6e-2             2.28e-2      --
#
# The fp8 gate sits between its clean maximum and its weakest RED almost
# exactly halfway (1.5x below, 1.5x above); the bf16 one leaves the same
# clean headroom with its weakest RED 2.6x above it. A gate looser than
# this passed the causal mutant (0.022 under the first cut's 3e-2), which
# is why the fp8 bound is measured against the fp8 arms and not carried
# over from bf16's.
_LOGITS_TOL = {
    DType.bfloat16: 1e-2,
    DType.float8_e4m3fn: 1.5e-2,
}


@pytest.mark.compile_on_gpu(
    "the fused MLA store and attention kernels bind as declared custom ops"
)
@pytest.mark.parametrize("cache_dtype", _CACHE_DTYPES)
@pytest.mark.parametrize(
    ("anchor_position", "num_devices"),
    [
        pytest.param(10, 1, id="anchor10_tp1"),
        pytest.param(100, 1, id="anchor100_tp1"),
        pytest.param(100, 2, id="anchor100_tp2"),
    ],
)
def test_matches_the_reference(
    anchor_position: int, num_devices: int, cache_dtype: DType
) -> None:
    config, weights, head, taps, anchor_token = _tiny_instance(anchor_position)

    got_logits, got_drafts = _run_module(
        config,
        weights,
        head,
        taps,
        anchor_token,
        anchor_position,
        num_devices,
        cache_dtype,
    )

    torch_weights = {
        name: torch.from_numpy(value) for name, value in weights.items()
    }
    reference = K3DSparkReference(
        TINY_DRAFT_CONFIG,
        torch_weights,
        torch.from_numpy(head),
        dtype=torch.float32,
    )
    want_logits = _reference_logits(
        reference, taps, anchor_token, anchor_position
    )
    want_drafts = reference.greedy_block(
        torch.from_numpy(_round_to_bf16(taps)),
        torch.arange(anchor_position),
        anchor_token,
        anchor_position,
    ).numpy()

    rel = rel_error(got_logits, want_logits)
    print(
        f"K3 DSpark module vs reference logits rel_error={rel}"
        f" at anchor_position={anchor_position}, {num_devices} device(s),"
        f" {cache_dtype} latent"
    )
    assert got_logits.shape == want_logits.shape
    # The gate per arm: bf16's prefill arm and fp8's fold both sit under
    # their own bound on the clean runs, and every RED below must clear its
    # dtype's gate. A mis-seeded markov chain leaves these base logits
    # clean but breaks the draft-equality assert below.
    assert rel < _LOGITS_TOL[cache_dtype], (
        f"K3 DSpark numerics at {num_devices} device(s), {cache_dtype}"
        f" latent: rel_error={rel}"
    )
    assert np.array_equal(got_drafts[0], want_drafts), (
        f"module drafts {got_drafts[0].tolist()} != reference"
        f" {want_drafts.tolist()}"
    )
    # The Markov bias must be load-bearing in this instance, or the draft
    # equality above would pass with the bias head dropped: the chain's
    # picks must disagree with the unbiased argmax of the module's own
    # logits somewhere.
    unbiased = np.argmax(got_logits.reshape(config.block_size, -1), axis=-1)
    assert not np.array_equal(got_drafts[0], unbiased), (
        "the markov bias never flipped an argmax in this instance"
    )


class _CausalInsteadOfNull:
    """Presents CAUSAL_MASK wherever the drafter's code names NULL_MASK.

    The block call reads ``MHAMaskVariant.NULL_MASK`` off its module's
    global, so swapping the symbol the test patches in flips the mask the
    attention op binds -- the block-diffusion mask mis-wired to the causal
    one every other attention path uses.
    """

    NULL_MASK = MHAMaskVariant.CAUSAL_MASK


# Every RED runs at the far anchor, the position whose long context most
# dilutes a block-sized defect and so gives each mutant its weakest
# signal: a gate the far RED clears cannot be loose. Both latent dtypes
# get the dropped-store control; the fold arm additionally gets the
# causal mask (its NULL_MASK routing is this lane's own claim, proven at
# kernel level by the null-mask suite), and the prefill arm the store's
# gamma (the in-cache gain this lane newly relies on).
_REDS = [
    pytest.param(DType.bfloat16, "dropped_store", id="bf16_dropped_store"),
    pytest.param(DType.float8_e4m3fn, "dropped_store", id="fp8_dropped_store"),
    pytest.param(DType.float8_e4m3fn, "causal_mask", id="fp8_causal_mask"),
    pytest.param(
        DType.bfloat16, "ones_gamma_store", id="bf16_ones_gamma_store"
    ),
]


@pytest.mark.compile_on_gpu(
    "the fused MLA store and attention kernels bind as declared custom ops"
)
@pytest.mark.parametrize(("cache_dtype", "mutant"), _REDS)
def test_red_control_mutants_diverge(
    cache_dtype: DType,
    mutant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each sabotage must fail the gate its clean run passes.

    The same module, weights and taps as the clean run, with one thing
    broken per case, so a green oracle is proven able to catch the defect
    class the mutant stands for: a store that silently stops storing, a
    block-diffusion mask mis-wired to causal, and a context store that
    drops its layernorm gain. A mutant that passed the gate would mean the
    oracle cannot see that defect.
    """
    if mutant == "causal_mask":
        monkeypatch.setattr(
            dspark_module, "MHAMaskVariant", _CausalInsteadOfNull
        )
    config, weights, head, taps, anchor_token = _tiny_instance(100)
    got_logits, _ = _run_module(
        config,
        weights,
        head,
        taps,
        anchor_token,
        100,
        1,
        cache_dtype,
        store_context=mutant != "dropped_store",
        store_gamma_ones=mutant == "ones_gamma_store",
    )
    torch_weights = {
        name: torch.from_numpy(value) for name, value in weights.items()
    }
    reference = K3DSparkReference(
        TINY_DRAFT_CONFIG,
        torch_weights,
        torch.from_numpy(head),
        dtype=torch.float32,
    )
    want_logits = _reference_logits(reference, taps, anchor_token, 100)
    rel = rel_error(got_logits, want_logits)
    print(f"RED {mutant} at {cache_dtype} latent: rel={rel}")
    assert rel >= _LOGITS_TOL[cache_dtype], (
        f"the {mutant} mutant passed the {cache_dtype} numerics gate"
        f" (rel={rel}); the oracle cannot catch it"
    )


@pytest.fixture(scope="module")
def golden() -> dict[str, torch.Tensor]:
    return load_file(GOLDEN_DIR / "golden_reference.safetensors")


@pytest.fixture(scope="module")
def golden_reference(golden: dict[str, torch.Tensor]) -> K3DSparkReference:
    with open(GOLDEN_DIR / "golden_reference_config.json") as f:
        config = json.load(f)
    # The weights are regenerated from the recorded seed, not read back
    # from the file: the checkpoint generator's own draw reproduces bit for
    # bit the weights the recorded outputs were computed from.
    weights = draw_state_dict(config, seed=int(golden["weight_seed"]))
    return K3DSparkReference(config, weights, golden["lm_head"])


@pytest.mark.parametrize("case", ["near", "far"])
def test_golden_reference_reproduces_torchspec(
    case: str,
    golden: dict[str, torch.Tensor],
    golden_reference: K3DSparkReference,
) -> None:
    """The reference must replay TorchSpec's recorded outputs exactly enough
    to reproduce the greedy block bit for bit at both positions."""
    anchor_position = int(golden[f"anchor_position_{case}"])
    anchor_token = int(golden["anchor_token"])
    context_hidden = golden["context_hidden"]
    context_positions = golden["context_positions"]

    expected_base = golden[f"expected_base_logits_{case}"]
    base = golden_reference.draft_logits(
        context_hidden, context_positions, anchor_token, anchor_position
    )
    assert torch.allclose(
        base,
        expected_base,
        rtol=0.0,
        atol=GOLDEN_LOGITS_ATOL,
    ), (base - expected_base).abs().max()

    biased = golden_reference.draft_logits(
        context_hidden,
        context_positions,
        anchor_token,
        anchor_position,
        prev_tokens=golden["prev_tokens"],
    )
    assert torch.allclose(
        biased,
        golden[f"expected_biased_logits_{case}"],
        rtol=0.0,
        atol=GOLDEN_LOGITS_ATOL,
    )

    drafted = golden_reference.greedy_block(
        context_hidden, context_positions, anchor_token, anchor_position
    )
    assert torch.equal(drafted, golden[f"expected_greedy_{case}"])
