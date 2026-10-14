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
"""Drives the fused MiMo-V2 DFlash graph and the MiMo-V2 base graphs by hand.

The graphs come from the production modules. The caches are paged pools whose
pages the caller assigns, so a test can give a request another request's pages
the way a prefix-cache hit does, and read any position of the drafter's cache
back. The tests run it on a random-weight model; the node checks on the real
checkpoint.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.engine import Model as CompiledModel
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn.comm import Signals
from max.nn.kv_cache import (
    PACKED_PAGE_STRIDE,
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAKVCacheParams,
    MultiKVCacheParams,
    padded_lut_cols,
)
from max.nn.layer import Module
from max.nn.sampling import AcceptanceSampler
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.dflash_mimo_v2 import (
    DFlashMiMoV2,
    DFlashMiMoV2Config,
)
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    MASK_EMBEDDING,
)
from max.pipelines.architectures.mimo_v2.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
    attention_head_dim,
    layer_types,
)
from max.pipelines.architectures.unified_dflash_mimo_v2 import (
    UnifiedDflashMiMoV2,
    UnifiedDflashMiMoV2Spec,
    fused_graph,
    prefixed_context_writer,
)
from max.pipelines.architectures.unified_dflash_mimo_v2.model_config import (
    DRAFT,
)
from max.pipelines.lib import SpeculativeConfig
from max.pipelines.lib.pipeline_variants.structured_output_overlap import (
    StructuredOutputOverlapState,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.weights._fp8 import e4m3fn_lut
from test_common.mef_precompile import init_from_mef, mefs_from_env
from transformers.configuration_utils import PretrainedConfig

PAGE_SIZE = 128

SAMPLEABLE = 1000
"""The tiny model's tokenizer size: ids from here to its vocab's 1,024 stand
in for the ``lm_head`` padding rows."""

MEF_RLOCATIONS = "MIMO_DFLASH_MEF_RLOCATIONS"
"""The environment variable naming the graphs compiled ahead on a CPU."""

PRECOMPILED_DEVICES = 2

F32 = npt.NDArray[np.float32]

TestMutation = Literal["accept_one_extra"]
"""A defect to build into the fused graph, to prove a check can fail."""


def bf16_bits(x: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Rounds to the nearest BFloat16 and returns its bits."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def to_f32(buffer: Buffer) -> F32:
    """A host copy of ``buffer`` as float32, whatever its float dtype."""
    host = buffer.to(CPU())
    if host.dtype == DType.bfloat16:
        bits = np.from_dlpack(host.view(DType.uint16)).astype(np.uint32)
        return (bits << 16).view(np.float32)
    return np.from_dlpack(host).astype(np.float32)


def weight_data(name: str, dtype: DType, array: np.ndarray) -> WeightData:
    buffer = Buffer.from_numpy(np.ascontiguousarray(array))
    if dtype != DType.from_numpy(array.dtype):
        buffer = buffer.view(dtype)
    return WeightData(buffer, name, dtype, Shape(array.shape))


def random_state(
    module: Module, rng: np.random.Generator
) -> dict[str, WeightData]:
    """Random weights for every weight ``module`` declares, sized to keep
    activations O(1) through a pre-norm residual stack."""
    fp8 = e4m3fn_lut()
    small_codes = np.flatnonzero(np.isfinite(fp8) & (np.abs(fp8) <= 2))
    fp8_rms = float(np.sqrt(np.mean(fp8[small_codes] ** 2)))
    state = {}
    for name, weight in module.raw_state_dict().items():
        assert isinstance(weight, Weight)
        shape = tuple(int(d) for d in weight.shape)
        dtype = weight.dtype
        if dtype == DType.float8_e4m3fn:
            array: np.ndarray = rng.choice(small_codes, shape).astype(np.uint8)
        elif dtype == DType.float8_e8m0fnu:
            # 2^-6: an E2M1 code's RMS is about 2.9, so rows of a few hundred
            # inputs stay near unit scale.
            array = np.full(shape, 121, np.uint8)
        elif dtype == DType.uint8:
            array = rng.integers(0, 256, shape, dtype=np.uint8)
        elif dtype == DType.float32:
            if name.endswith("weight_scale"):
                k = shape[1] * 128
                array = np.full(shape, 1 / (fp8_rms * math.sqrt(k)), np.float32)
            elif len(shape) == 2:
                array = rng.standard_normal(shape, np.float32) / math.sqrt(
                    shape[1]
                )
            else:
                array = 0.1 * rng.standard_normal(shape, np.float32)
        else:
            assert dtype == DType.bfloat16, (name, dtype)
            if "norm" in name:
                values = 1 + 0.05 * rng.standard_normal(shape)
            elif "sink" in name:
                values = rng.standard_normal(shape)
            elif "embed" in name or name == "mask_embedding":
                values = rng.standard_normal(shape)
            elif len(shape) == 2:
                values = rng.standard_normal(shape) / math.sqrt(shape[1])
            else:
                values = 0.1 * rng.standard_normal(shape)
            array = bf16_bits(values)
        state[name] = weight_data(name, dtype, array)
    return state


def tiny_target_hf(num_layers: int = 3, experts: int = 32) -> PretrainedConfig:
    """A small MiMo-V2 in the NVFP4 export's config: full attention with a
    dense MLP, then SWA-128 with sinks and MoE, then full attention and MoE.

    The router kernel runs a thread per expert in whole warps, so the expert
    count is a multiple of 32.
    """
    pattern = [0] + [1, 0] * ((num_layers - 1) // 2)
    pattern += [1] * (num_layers - len(pattern))
    moe = [0] + [1] * (num_layers - 1)
    return PretrainedConfig(
        vocab_size=1024,
        hidden_size=512,
        num_hidden_layers=num_layers,
        layernorm_epsilon=1e-6,
        hybrid_layer_pattern=pattern,
        moe_layer_freq=moe,
        num_attention_heads=8,
        swa_num_attention_heads=8,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        partial_rotary_factor=0.334,
        rope_theta=10000000.0,
        swa_rope_theta=10000.0,
        sliding_window=128,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        intermediate_size=1024,
        moe_intermediate_size=512,
        n_routed_experts=experts,
        num_experts_per_tok=8,
        n_shared_experts=None,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
        attention_projection_layout="fused_qkv",
        conversion_metadata={"qkv_layout": "global_q_k_v"},
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": {
                f"model.layers.{layer}.mlp.experts.{expert}.{proj}": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
                for layer in range(num_layers)
                if moe[layer]
                for expert in range(experts)
                for proj in ("gate_proj", "up_proj", "down_proj")
            },
        },
    )


def tiny_dflash_config(target_layer_ids: Sequence[int]) -> dict[str, Any]:
    """A small drafter's ``config.json``, as ``from_dflash_config`` reads it."""
    return {
        "architectures": ["DFlashDraftModel"],
        "hidden_size": 512,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "v_head_dim": 128,
        "partial_rotary_factor": 0.5,
        "block_size": 8,
        "dflash_config": {
            "target_layer_ids": list(target_layer_ids),
            "mask_token_id": 1000,
            "block_size": 8,
            "attention_value_scale": 0.612,
            "attention_sink_bias": True,
        },
        "layer_types": ["sliding_attention"] * 2,
        "sliding_window": 1024,
        "use_sliding_window": True,
        "is_causal": False,
        "num_target_layers": 3,
        "target_hidden_size": 512,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "add_swa_attention_sink_bias": True,
        "tie_word_embeddings": False,
    }


def target_kv_params(
    hf: PretrainedConfig,
    devices: Sequence[DeviceRef],
    *,
    num_draft_tokens: int | None = None,
) -> MultiKVCacheParams:
    """The target's ``{sliding, full}`` tree; with ``num_draft_tokens`` the
    speculative graph's, whose groups also take a draft dispatch buffer."""
    groups = layer_types(hf)

    def group(heads: int, layers: int, window: int | None) -> MHAKVCacheParams:
        return MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=heads,
            head_dim=attention_head_dim(hf),
            num_layers=layers,
            devices=list(devices),
            page_size=PAGE_SIZE,
            window_size=window,
            speculative_method=None if num_draft_tokens is None else "dflash",
            num_draft_tokens=num_draft_tokens or 0,
        )

    return MultiKVCacheParams.from_params(
        {
            SLIDING: group(
                hf.swa_num_key_value_heads,
                groups.count(SLIDING),
                hf.sliding_window,
            ),
            FULL: group(hf.num_key_value_heads, groups.count(FULL), None),
        }
    )


def drafter_kv_params(
    dflash: Mapping[str, Any],
    devices: Sequence[DeviceRef],
    *,
    speculative: bool,
) -> MHAKVCacheParams:
    """The drafter's context cache, the tail group of every tree with one."""
    block = int(dflash["block_size"])
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=int(dflash["num_key_value_heads"]),
        head_dim=int(dflash["head_dim"]),
        num_layers=int(dflash["num_hidden_layers"]),
        devices=list(devices),
        page_size=PAGE_SIZE,
        window_size=int(dflash["sliding_window"]),
        speculative_method="dflash" if speculative else None,
        num_draft_tokens=block if speculative else 0,
    )


def sampleable_bitmask(
    batch: int, positions: int, vocab: int, sampleable: int
) -> npt.NDArray[np.int32]:
    """The packed bitmask that allows exactly the ids below ``sampleable``.

    Bit ``t % 32`` of word ``t // 32`` is token ``t``, as the kernel reads it.
    """
    words = np.zeros(-(-vocab // 32), np.uint32)
    words[: sampleable // 32] = 0xFFFFFFFF
    if sampleable % 32:
        words[sampleable // 32] = (1 << (sampleable % 32)) - 1
    return np.ascontiguousarray(
        np.broadcast_to(words.view(np.int32), (batch, positions, words.size))
    )


class Pages:
    """Hands out page ids, never reusing one."""

    def __init__(self) -> None:
        self.next = 0

    def take(self, positions: int) -> list[int]:
        count = -(-positions // PAGE_SIZE)
        pages = list(range(self.next, self.next + count))
        self.next += count
        return pages


class PagedTree:
    """One page pool per KV group and device.

    The pools hold no speculative state, so a speculative graph and a base
    graph can share them; which dispatch buffers a step passes comes from the
    params each graph was built with.
    """

    def __init__(
        self,
        groups: Mapping[str, KVCacheParams],
        devices: Sequence[Device],
        num_pages: int,
    ) -> None:
        self.devices = list(devices)
        self.num_pages = num_pages
        # The page past the pool is the null page every unassigned lookup
        # column names.
        self.blocks = {
            name: [
                Buffer.zeros(
                    [num_pages + 1, *params.shape_per_block],
                    params.dtype,
                    device,
                )
                for device in self.devices
            ]
            for name, params in groups.items()
        }

    def inputs(
        self,
        groups: Mapping[str, KVCacheParams],
        pages: Sequence[Sequence[int]],
        cache_lengths: Sequence[int],
        row_lengths: Sequence[int],
        reach: Mapping[str, int] | None = None,
    ) -> list[Buffer]:
        """The flattened KV inputs of every group, in ``groups``' order.

        Args:
            groups: The params the graph declared its KV inputs from.
            pages: Per request, its pages in position order.
            cache_lengths: Per request, the positions already cached.
            row_lengths: Per request, the rows this step writes.
            reach: Per group, how far past a request's rows the step reads,
                for a block written beyond them.
        """
        flat: list[Buffer] = []
        cols = padded_lut_cols(max(len(p) for p in pages))
        table = np.full((len(pages), cols), self.num_pages, np.uint32)
        for row, request in enumerate(pages):
            table[row, : len(request)] = request
        for name, params in groups.items():
            extra = (reach or {}).get(name, 0)
            max_rows = max(row_lengths)
            ends = [
                c + r + extra
                for c, r in zip(cache_lengths, row_lengths, strict=True)
            ]
            assert all(
                end <= len(p) * PAGE_SIZE
                for end, p in zip(ends, pages, strict=True)
            ), "a request reads past its pages"
            max_cache = max(ends)
            key = params.resolve_attn_key(len(pages), max_rows, max_cache)
            draft_key = (
                params.resolve_attn_key(
                    len(pages), params.num_draft_tokens_per_step, max_cache
                )
                if params.speculative_method is not None
                else None
            )
            for device, blocks in zip(
                self.devices, self.blocks[name], strict=True
            ):
                flat += tree.leaves(
                    KVCacheInputsPerDevice(
                        kv_blocks=blocks,
                        cache_lengths=Buffer.from_numpy(
                            np.array(cache_lengths, np.uint32)
                        ).to(device),
                        lookup_table=Buffer.from_numpy(table).to(device),
                        max_prompt_length=Buffer.from_numpy(
                            np.array([max_rows], np.uint32)
                        ),
                        max_cache_length=Buffer.from_numpy(
                            np.array([max_cache], np.uint32)
                        ),
                        page_stride=Buffer.from_numpy(
                            np.array([PACKED_PAGE_STRIDE], np.int64)
                        ),
                        attention_dispatch_metadata=key.pack_into_buffer(
                            device, max_cache
                        ),
                        draft_attention_dispatch_metadata=(
                            draft_key.pack_into_buffer(device, max_cache)
                            if draft_key is not None
                            else None
                        ),
                    )
                )
        return flat

    def export_pages(
        self, pages: Sequence[int]
    ) -> dict[str, list[npt.NDArray[np.uint8]]]:
        """The raw bytes of ``pages`` in every group, per device."""
        out: dict[str, list[npt.NDArray[np.uint8]]] = {}
        for name, blocks in self.blocks.items():
            out[name] = []
            for b in blocks:
                host = np.from_dlpack(b.to(CPU()).view(DType.uint8))
                out[name].append(host[list(pages)].astype(np.uint8))
        return out

    def import_pages(
        self,
        pages: Sequence[int],
        data: Mapping[str, Sequence[npt.NDArray[np.uint8]]],
    ) -> None:
        """Writes :meth:`export_pages`' bytes into ``pages``, as a prefix
        cache restores a hit."""
        for name, blocks in self.blocks.items():
            for d, (b, device) in enumerate(
                zip(blocks, self.devices, strict=True)
            ):
                host = np.from_dlpack(b.to(CPU()).view(DType.uint8)).copy()
                host[list(pages)] = data[name][d]
                blocks[d] = Buffer.from_numpy(host).view(b.dtype).to(device)

    def read(self, group: str, pages: Sequence[int], positions: range) -> F32:
        """Returns ``[2, layers, len(positions), heads, head_dim]`` K and V,
        every device's heads side by side."""
        per_device = []
        for blocks in self.blocks[group]:
            host = to_f32(blocks)
            rows = [
                host[pages[p // PAGE_SIZE], :, :, p % PAGE_SIZE]
                for p in positions
            ]
            per_device.append(np.stack(rows, axis=2))
        return np.concatenate(per_device, axis=-2)


@dataclass(frozen=True)
class Sampling:
    """One request's sampling parameters; temperature 0 is greedy."""

    seed: int = 0
    """The request's base seed; a step keys its draws off it plus the tokens
    the request has generated."""
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0


GREEDY = Sampling()


@dataclass
class Row:
    """One request's part of a step."""

    tokens: list[int]
    """A prefill chunk, or the anchor of a decode step."""
    cache_length: int
    pages: list[int]
    drafts: list[int] = field(default_factory=list)
    """The previous step's proposals, empty on a prefill."""
    sampling: Sampling = GREEDY
    generated: int = 0
    """Tokens the request has committed so far."""


def _tail(
    rows: Sequence[Sampling], generated: Sequence[int], device: Device
) -> tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer]:
    """The sampling inputs: per-row seed, temperature, top-k and top-p, and
    the batch's largest top-k and smallest top-p on the host."""
    seeds = [
        (r.seed + g) % (1 << 64) for r, g in zip(rows, generated, strict=True)
    ]
    top_k = np.array([r.top_k for r in rows], np.int64)
    top_p = np.array([r.top_p for r in rows], np.float32)
    return (
        Buffer.from_numpy(np.array(seeds, np.uint64)).to(device),
        Buffer.from_numpy(
            np.array([r.temperature for r in rows], np.float32)
        ).to(device),
        Buffer.from_numpy(top_k).to(device),
        Buffer.from_numpy(np.array(top_k.max(), np.int64)),
        Buffer.from_numpy(top_p).to(device),
        Buffer.from_numpy(np.array(top_p.min(), np.float32)),
    )


class RecordingSampler(AcceptanceSampler):
    """Forwards to an acceptance sampler and keeps the logits it verified
    against, optionally accepting one draft past the first rejection.

    With ``accept_one_extra`` the committed token is the target's own
    prediction after the extra draft, so the output stays self-consistent
    except at that draft, which the target did not choose.
    """

    def __init__(
        self, sampler: AcceptanceSampler, *, accept_one_extra: bool = False
    ) -> None:
        # Only forwards, so the wrapped sampler keeps the configuration.
        self.sampler = sampler
        self.accept_one_extra = accept_one_extra
        self.target_logits: TensorValue | None = None

    def __call__(
        self,
        draft_tokens: TensorValue,
        target_logits: TensorValue,
        **kwargs: Any,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        self.target_logits = target_logits
        num_accepted, recovered, bonus = self.sampler(
            draft_tokens, target_logits, **kwargs
        )
        if not self.accept_one_extra:
            return num_accepted, recovered, bonus
        device = draft_tokens.device
        steps = _shape_to_scalar(
            draft_tokens.shape[1], device, dtype=num_accepted.dtype
        ).broadcast_to(num_accepted.shape)
        one = ops.constant(1, num_accepted.dtype, device=device)
        return ops.min(num_accepted + one, steps), recovered, bonus


class VerifyLogitsFused(UnifiedDflashMiMoV2):
    """The fused module with the logits acceptance verified against as a
    fourth output: ``[batch * (K + 1), vocab]`` float32, or
    ``[batch, vocab]`` on a prefill."""

    def __init__(
        self,
        spec: UnifiedDflashMiMoV2Spec,
        mask_embedding: WeightData,
        enable_structured_output: bool,
        test_mutation: TestMutation | None = None,
    ) -> None:
        super().__init__(spec, mask_embedding, enable_structured_output)
        self.recording = RecordingSampler(
            self.acceptance_sampler,
            accept_one_extra=test_mutation == "accept_one_extra",
        )
        self.acceptance_sampler = self.recording

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[TensorValue, ...]:
        outputs = super().__call__(*args, **kwargs)
        assert self.recording.target_logits is not None
        return (*outputs, self.recording.target_logits)


def fused_runner_graph(
    spec: UnifiedDflashMiMoV2Spec,
    target_state: Mapping[str, WeightData],
    draft_state: Mapping[str, WeightData],
    mask_embedding: WeightData,
    test_mutation: TestMutation | None = None,
) -> tuple[Graph, dict[str, Any]]:
    """The graph :class:`FusedRunner` runs, and its weights registry."""
    nn_model = VerifyLogitsFused(
        spec,
        mask_embedding,
        enable_structured_output=True,
        test_mutation=test_mutation,
    )
    nn_model.load_state_dict(
        {
            **{f"target.{k}": v for k, v in target_state.items()},
            **{f"{DRAFT}.{k}": v for k, v in draft_state.items()},
        },
        weight_alignment=1,
        strict=True,
    )
    registry = nn_model.state_dict(auto_initialize=False)
    target_kv = spec.target.kv_params
    assert isinstance(target_kv, MultiKVCacheParams)
    kv_params = MultiKVCacheParams.from_params(
        {"target": target_kv, DRAFT: spec.draft.kv_params}
    )
    return fused_graph(nn_model, kv_params), registry


def _load(
    session: InferenceSession,
    graph: Graph,
    registry: Mapping[str, Any],
    mef: Path | None,
) -> CompiledModel:
    if mef is None:
        return session.load(graph, weights_registry=registry)
    return init_from_mef(session, mef, registry)


def precompiled_mef(name: str, devices: Sequence[Device]) -> Path | None:
    """The MEF of :func:`named_graph` ``name`` compiled ahead on a CPU, or
    ``None`` to compile it here."""
    if MEF_RLOCATIONS not in os.environ or len(devices) != PRECOMPILED_DEVICES:
        return None
    return mefs_from_env(MEF_RLOCATIONS)[f"{name}.mef"]


@dataclass
class FusedStep:
    """What one fused step returned, per request."""

    num_accepted: npt.NDArray[np.int64]
    next_tokens: npt.NDArray[np.int64]
    next_drafts: npt.NDArray[np.int64]
    logits: F32
    """The verify logits: per request, one row per verified position."""


class FusedRunner:
    """The fused graph, compiled with the grammar bitmask and its verify
    logits as an extra output."""

    def __init__(
        self,
        spec: UnifiedDflashMiMoV2Spec,
        target_state: Mapping[str, WeightData],
        draft_state: Mapping[str, WeightData],
        mask_embedding: WeightData,
        devices: Sequence[Device],
        session: InferenceSession,
        *,
        max_batch_size: int = 16,
        test_mutation: TestMutation | None = None,
        bitmask_allows_padding: bool = False,
        mef: Path | None = None,
    ) -> None:
        self.devices = list(devices)
        self.spec = spec
        self.k = spec.num_speculative_tokens
        self.block = spec.draft.block_size
        graph, registry = fused_runner_graph(
            spec, target_state, draft_state, mask_embedding, test_mutation
        )
        self.model = _load(session, graph, registry, mef)
        target_kv = spec.target.kv_params
        assert isinstance(target_kv, MultiKVCacheParams)
        self.groups: dict[str, KVCacheParams] = {}
        for name in (SLIDING, FULL):
            leaf = target_kv.children[name]
            assert isinstance(leaf, KVCacheParams)
            self.groups[name] = leaf
        self.groups[DRAFT] = spec.draft.kv_params
        self.signals = Signals.allocate(self.devices)
        self.vocab = spec.target.vocab_size
        # Allowing the padding rows leaves the graph's own mask alone
        # keeping them out.
        self.allowed = (
            self.vocab if bitmask_allows_padding else spec.sampleable_vocab_size
        )
        self.bitmask = StructuredOutputOverlapState(
            self.devices[0],
            CPU(),
            max_batch_size,
            [1, self.k + 1],
            self.vocab,
        )

    def step(self, caches: PagedTree, rows: Sequence[Row]) -> FusedStep:
        """Runs one step; every row carries K drafts, or every row none."""
        drafts = {len(r.drafts) for r in rows}
        assert drafts in ({0}, {self.k}), drafts
        steps = drafts.pop()
        device = self.devices[0]
        batch = len(rows)
        tokens = np.concatenate([np.array(r.tokens, np.int64) for r in rows])
        offsets = np.cumsum([0] + [len(r.tokens) for r in rows]).astype(
            np.uint32
        )
        draft_tokens = np.array([r.drafts for r in rows], np.int64).reshape(
            batch, steps
        )
        positions = steps + 1
        self.bitmask.prime(
            sampleable_bitmask(batch, positions, self.vocab, self.allowed)
        )
        pinned, scratch = self.bitmask.get_input_views(batch, positions)
        outputs = self.model.execute(
            Buffer.from_numpy(tokens).to(device),
            Buffer.from_numpy(offsets).to(device),
            Buffer.from_numpy(np.array([positions], np.int64)),
            *self.signals,
            *caches.inputs(
                self.groups,
                [r.pages for r in rows],
                [r.cache_length for r in rows],
                [len(r.tokens) + len(r.drafts) for r in rows],
                reach={DRAFT: self.block},
            ),
            Buffer.from_numpy(draft_tokens).to(device),
            *_tail(
                [r.sampling for r in rows], [r.generated for r in rows], device
            ),
            pinned,
            self.bitmask.wait_payload,
            scratch,
        )
        num_accepted, next_tokens, next_drafts, logits = (
            o.to(CPU()) for o in outputs
        )
        assert isinstance(num_accepted, Buffer)
        assert isinstance(next_tokens, Buffer)
        assert isinstance(next_drafts, Buffer)
        assert isinstance(logits, Buffer)
        return FusedStep(
            num_accepted=np.from_dlpack(num_accepted).astype(np.int64),
            next_tokens=np.from_dlpack(next_tokens).astype(np.int64).ravel(),
            next_drafts=np.from_dlpack(next_drafts).astype(np.int64),
            logits=to_f32(logits).reshape(batch, positions, self.vocab),
        )


def base_graph(
    target: MiMoV2Config,
    target_state: Mapping[str, WeightData] | None,
    *,
    drafter: DFlashMiMoV2Config | None = None,
    draft_state: Mapping[str, WeightData] | None = None,
) -> tuple[Graph, dict[str, Any], dict[str, KVCacheParams]]:
    """The MiMo-V2 base graph returning every row's logits, in the input
    order ``MiMoV2Model`` builds; with ``drafter`` the base-ctx build.

    Without weights, the weights are only named, enough to read the
    signature. Returns the graph, its weights registry and its KV groups.
    """
    config = dataclasses.replace(
        target, return_logits=ReturnLogits.ALL, target_layer_ids=None
    )
    refs = list(config.devices)
    target_kv = config.kv_params
    assert isinstance(target_kv, MultiKVCacheParams)
    groups: dict[str, KVCacheParams] = {}
    for name in (SLIDING, FULL):
        leaf = target_kv.children[name]
        assert isinstance(leaf, KVCacheParams)
        groups[name] = leaf
    hook = None
    writer_registry: dict[str, Any] = {}
    if drafter is not None:
        config.target_layer_ids = list(drafter.target_layer_ids)
        hook, writer_registry = prefixed_context_writer(drafter, draft_state)
        groups[DRAFT] = drafter.kv_params
    model = MiMoV2(config, tap_hook=hook)
    registry: dict[str, Any] = {}
    if target_state is not None:
        model.load_state_dict(
            dict(target_state), weight_alignment=1, strict=True
        )
        registry = {
            **model.state_dict(auto_initialize=False),
            **writer_registry,
        }
    else:
        for name, weight in model.raw_state_dict().items():
            weight.name = name
    kv_params = MultiKVCacheParams.from_params(groups)
    n = len(refs)
    input_types = [
        TensorType(DType.int64, ["total_seq_len"], device=refs[0]),
        TensorType(DType.int64, ["return_n_logits"], device=DeviceRef.CPU()),
        *(
            TensorType(DType.uint32, ["input_row_offsets_len"], device=r)
            for r in refs
        ),
        *Signals(devices=refs).input_types(),
        *tree.leaves(kv_params.get_symbolic_inputs()),
    ]
    with Graph("mimo_v2", input_types=input_types) as graph:
        tokens, return_n_logits, *rest = graph.inputs
        sliding, full, *tail = kv_params.unflatten_basic_kv_tree(
            iter(rest[2 * n :])
        )
        outputs = model(
            tokens=tokens.tensor,
            signal_buffers=[v.buffer for v in rest[n : 2 * n]],
            sliding_kv_collections=sliding,
            full_kv_collections=full,
            return_n_logits=return_n_logits.tensor,
            input_row_offsets=[v.tensor for v in rest[:n]],
            tail_kv_collections=tail[0] if tail else None,
        )
        graph.output(*outputs)
    return graph, registry, groups


class BaseRunner:
    """The MiMo-V2 base graph returning every row's logits, optionally with
    the drafter's context writer (base-ctx)."""

    def __init__(
        self,
        target: MiMoV2Config,
        target_state: Mapping[str, WeightData],
        devices: Sequence[Device],
        session: InferenceSession,
        *,
        drafter: DFlashMiMoV2Config | None = None,
        draft_state: Mapping[str, WeightData] | None = None,
        mef: Path | None = None,
    ) -> None:
        self.devices = list(devices)
        self.vocab = target.vocab_size
        graph, registry, self.groups = base_graph(
            target, target_state, drafter=drafter, draft_state=draft_state
        )
        self.model = _load(session, graph, registry, mef)
        self.signals = Signals.allocate(self.devices)

    def step(self, caches: PagedTree, rows: Sequence[Row]) -> F32:
        """Runs one step; returns ``[rows, vocab]`` logits for every row."""
        tokens = np.concatenate([np.array(r.tokens, np.int64) for r in rows])
        offsets = Buffer.from_numpy(
            np.cumsum([0] + [len(r.tokens) for r in rows]).astype(np.uint32)
        )
        outputs = self.model.execute(
            Buffer.from_numpy(tokens).to(self.devices[0]),
            Buffer.from_numpy(np.array([1], np.int64)),
            *(offsets.to(d) for d in self.devices),
            *self.signals,
            *caches.inputs(
                self.groups,
                [r.pages for r in rows],
                [r.cache_length for r in rows],
                [len(r.tokens) for r in rows],
            ),
        )
        # (last-token logits, all logits, offsets).
        logits = outputs[1]
        assert isinstance(logits, Buffer)
        return to_f32(logits)


@dataclass
class Model:
    """A target, a drafter and their weights, at any device count."""

    hf: PretrainedConfig
    dflash: dict[str, Any]
    target_state: dict[str, WeightData]
    draft_state: dict[str, WeightData]
    mask_embedding: WeightData
    max_seq_len: int

    def target(
        self, devices: Sequence[DeviceRef], *, speculative: bool
    ) -> MiMoV2Config:
        """The target's config; the speculative graph's KV groups carry the
        block's draft count."""
        block = int(self.dflash["block_size"])
        return MiMoV2Config.from_huggingface_config(
            self.hf,
            devices=list(devices),
            kv_params=target_kv_params(
                self.hf,
                devices,
                num_draft_tokens=block if speculative else None,
            ),
            max_seq_len=self.max_seq_len + 4 * block,
        )

    def drafter(
        self, devices: Sequence[DeviceRef], *, speculative: bool
    ) -> DFlashMiMoV2Config:
        return DFlashMiMoV2Config.from_dflash_config(
            self.dflash,
            devices=list(devices),
            kv_params=drafter_kv_params(
                self.dflash, devices, speculative=speculative
            ),
            max_seq_len=self.max_seq_len + 4 * int(self.dflash["block_size"]),
        )

    def spec(
        self,
        devices: Sequence[DeviceRef],
        num_speculative_tokens: int,
        sampleable_vocab_size: int,
        *,
        greedy: bool = True,
    ) -> UnifiedDflashMiMoV2Spec:
        """The fused graph's spec; ``greedy=False`` accepts by rejection
        sampling under each row's own parameters."""
        return UnifiedDflashMiMoV2Spec(
            target=self.target(devices, speculative=True),
            draft=self.drafter(devices, speculative=True),
            speculative_config=SpeculativeConfig(
                speculative_method="dflash",
                num_speculative_tokens=num_speculative_tokens,
                use_greedy_acceptance=greedy,
            ),
            num_speculative_tokens=num_speculative_tokens,
            sampleable_vocab_size=sampleable_vocab_size,
        )


def tiny_configs(max_seq_len: int = 4096) -> Model:
    """The tiny target and drafter without weights, enough to build graphs."""
    hf = tiny_target_hf()
    dflash = tiny_dflash_config(list(range(hf.num_hidden_layers)))
    hidden = np.zeros(dflash["hidden_size"], np.uint16)
    return Model(
        hf=hf,
        dflash=dflash,
        target_state={},
        draft_state={},
        mask_embedding=weight_data("mask_embedding", DType.bfloat16, hidden),
        max_seq_len=max_seq_len,
    )


def tiny_model(seed: int = 0, max_seq_len: int = 4096) -> Model:
    """A random-weight target and drafter in the serving layouts."""
    rng = np.random.default_rng(seed)
    hf = tiny_target_hf()
    dflash = tiny_dflash_config(list(range(hf.num_hidden_layers)))
    one = [DeviceRef.GPU()]
    model = Model(
        hf=hf,
        dflash=dflash,
        target_state={},
        draft_state={},
        mask_embedding=weight_data(
            "mask_embedding", DType.bfloat16, np.zeros(1, np.uint16)
        ),
        max_seq_len=max_seq_len,
    )
    model.target_state = random_state(
        MiMoV2(model.target(one, speculative=False)), rng
    )
    # The correction bias alone picks each layer's experts, so two graphs
    # that compute a row in different chunks cannot route it differently: a
    # near-tie routing flip is a discrete difference no BF16 tolerance bounds.
    experts, top_k = hf.n_routed_experts, hf.num_experts_per_tok
    for name in list(model.target_state):
        if name.endswith("e_score_correction_bias"):
            bias = np.zeros(experts, np.float32)
            bias[rng.choice(experts, top_k, replace=False)] = 10.0
            model.target_state[name] = weight_data(name, DType.float32, bias)
    model.draft_state = random_state(
        DFlashMiMoV2(model.drafter(one, speculative=False)), rng
    )
    model.mask_embedding = model.draft_state.pop(MASK_EMBEDDING)
    return model


FUSED_GRAPHS: dict[str, tuple[int, bool, TestMutation | None]] = {
    "fused_k3": (3, True, None),
    "fused_k7": (7, True, None),
    "fused_k7_accept_one_extra": (7, True, "accept_one_extra"),
    "fused_k7_sampled": (7, False, None),
}
"""The tiny model's fused graphs: K, whether acceptance is greedy, and the
defect built in."""

BASE_GRAPHS = ("base", "base_ctx")
"""The tiny model's base graph, without and with the context writer."""


def named_graph(name: str) -> Graph:
    """Builds the tiny model's graph ``name`` over ``PRECOMPILED_DEVICES``
    GPUs, as :func:`fused_runner` and :func:`base_runner` build it."""
    model = tiny_model()
    refs = [DeviceRef.GPU(i) for i in range(PRECOMPILED_DEVICES)]
    if name in BASE_GRAPHS:
        graph, _, _ = base_graph(
            model.target(refs, speculative=False),
            model.target_state,
            drafter=(
                model.drafter(refs, speculative=False)
                if name == "base_ctx"
                else None
            ),
            draft_state=model.draft_state if name == "base_ctx" else None,
        )
        return graph
    k, greedy, test_mutation = FUSED_GRAPHS[name]
    graph, _ = fused_runner_graph(
        model.spec(refs, k, SAMPLEABLE, greedy=greedy),
        model.target_state,
        model.draft_state,
        model.mask_embedding,
        test_mutation,
    )
    return graph


def fused_runner(
    name: str,
    model: Model,
    devices: Sequence[Device],
    session: InferenceSession,
    *,
    target_state: Mapping[str, WeightData] | None = None,
    bitmask_allows_padding: bool = False,
) -> FusedRunner:
    """The tiny model's fused graph ``name``, precompiled when it can be."""
    k, greedy, test_mutation = FUSED_GRAPHS[name]
    refs = [DeviceRef.from_device(d) for d in devices]
    return FusedRunner(
        model.spec(refs, k, SAMPLEABLE, greedy=greedy),
        model.target_state if target_state is None else target_state,
        model.draft_state,
        model.mask_embedding,
        devices,
        session,
        test_mutation=test_mutation,
        bitmask_allows_padding=bitmask_allows_padding,
        mef=precompiled_mef(name, devices),
    )


def base_runner(
    name: str,
    model: Model,
    devices: Sequence[Device],
    session: InferenceSession,
) -> BaseRunner:
    """The tiny model's base graph ``name``, precompiled when it can be."""
    assert name in BASE_GRAPHS, name
    refs = [DeviceRef.from_device(d) for d in devices]
    writer = name == "base_ctx"
    return BaseRunner(
        model.target(refs, speculative=False),
        model.target_state,
        devices,
        session,
        drafter=model.drafter(refs, speculative=False) if writer else None,
        draft_state=model.draft_state if writer else None,
        mef=precompiled_mef(name, devices),
    )


def committed_token_violations(
    rows: Sequence[Row], step: FusedStep, sampleable: int
) -> list[str]:
    """Where a step committed anything but the target's own greedy choice.

    Row ``j`` of a request's verify logits predicts the token after its
    ``j``-th verified position, so the first ``a`` drafts must each be that
    row's argmax, bitmask applied, the committed token row ``a``'s, and the
    next draft must not match, or it too would have been accepted.
    """
    problems = []
    for i, row in enumerate(rows):
        target = step.logits[i][:, :sampleable].argmax(-1)
        a = int(step.num_accepted[i])
        if list(row.drafts[:a]) != target[:a].tolist():
            problems.append(
                f"row {i}: accepted {row.drafts[:a]}, target {target[:a]}"
            )
        if int(step.next_tokens[i]) != int(target[a]):
            problems.append(
                f"row {i}: committed {step.next_tokens[i]} after {a}"
                f" drafts, target {target[a]}"
            )
        if a < len(row.drafts) and row.drafts[a] == target[a]:
            problems.append(f"row {i}: stopped at {a} on a matching draft")
    return problems


@dataclass
class Generation:
    """Greedy speculative decoding's output, per request."""

    tokens: list[list[int]]
    """Committed output tokens, cut after a stop id or at the limit."""
    delivered: list[list[int]]
    """Per drafted verify step, the tokens it delivered, bonus included."""
    accepted: list[list[int]]
    """Per drafted verify step, the drafts it accepted."""
    violations: list[str]

    def acceptance_length(self) -> float:
        """Tokens per drafted verify step, pooled: a ratio of sums."""
        steps = sum(len(d) for d in self.delivered)
        return sum(sum(d) for d in self.delivered) / max(1, steps)


def generate(
    runner: FusedRunner,
    caches: PagedTree,
    pages: Pages,
    prompts: Sequence[Sequence[int]],
    max_new: int,
    *,
    stop_ids: Sequence[int] = (),
    guide: Sequence[Sequence[int]] | None = None,
    rng: np.random.Generator | None = None,
    sampling: Sequence[Sampling] | None = None,
) -> Generation:
    """Speculative decoding through the fused graph, one batch; greedy
    unless ``sampling`` gives each request its own parameters.

    With ``guide``, each step's drafts are the guide's next tokens, one
    replaced at random half the time, instead of the drafter's; that reaches
    every accepted count with a drafter that rarely agrees with its target.
    """
    k = runner.k
    reach = max_new + 2 * (k + 1) + 4 * runner.block
    params = list(sampling) if sampling is not None else [GREEDY] * len(prompts)
    rows = [
        Row(list(p), 0, pages.take(len(p) + reach), sampling=params[i])
        for i, p in enumerate(prompts)
    ]
    out = runner.step(caches, rows)
    sampleable = runner.spec.sampleable_vocab_size
    result = Generation(
        tokens=[[] for _ in prompts],
        delivered=[[] for _ in prompts],
        accepted=[[] for _ in prompts],
        violations=committed_token_violations(rows, out, sampleable),
    )
    lengths = [len(p) for p in prompts]
    anchors = [int(t) for t in out.next_tokens]
    drafts = out.next_drafts.tolist()
    live = []
    for i, token in enumerate(anchors):
        result.tokens[i].append(token)
        if token not in stop_ids and max_new > 1:
            live.append(i)
    while live:
        if guide is not None:
            assert rng is not None
            for i in live:
                done = len(result.tokens[i])
                proposal = list(guide[i][done : done + k])
                proposal += [0] * (k - len(proposal))
                if rng.random() < 0.5:
                    proposal[rng.integers(k)] = int(rng.integers(sampleable))
                drafts[i] = proposal
        step_rows = [
            Row(
                [anchors[i]],
                lengths[i],
                rows[i].pages,
                drafts[i],
                sampling=params[i],
                generated=len(result.tokens[i]),
            )
            for i in live
        ]
        out = runner.step(caches, step_rows)
        result.violations += committed_token_violations(
            step_rows, out, sampleable
        )
        still = []
        for j, i in enumerate(live):
            a = int(out.num_accepted[j])
            new = step_rows[j].drafts[:a] + [int(out.next_tokens[j])]
            delivered, stopped = 0, False
            for token in new:
                result.tokens[i].append(token)
                delivered += 1
                if token in stop_ids or len(result.tokens[i]) >= max_new:
                    stopped = True
                    break
            result.delivered[i].append(delivered)
            result.accepted[i].append(a)
            lengths[i] += 1 + a
            anchors[i] = new[-1]
            drafts[i] = out.next_drafts[j].tolist()
            if not stopped:
                still.append(i)
        live = still
    return result


def greedy(
    runner: BaseRunner,
    caches: PagedTree,
    pages: Pages,
    prompts: Sequence[Sequence[int]],
    tokens: int,
    sampleable: int,
    forced: Sequence[Sequence[int]] | None = None,
) -> tuple[list[list[int]], list[list[F32]]]:
    """Plain greedy decoding, one token a step, or teacher-forced on
    ``forced``; returns the tokens and the logits each was chosen from."""
    rows = [Row(list(p), 0, pages.take(len(p) + tokens + 1)) for p in prompts]
    out = runner.step(caches, rows)
    ends = np.cumsum([len(p) for p in prompts]) - 1
    step_logits = [out[e] for e in ends]
    generated: list[list[int]] = []
    logits: list[list[F32]] = []
    for _ in range(tokens):
        logits.append(step_logits)
        generated.append(
            [
                int(forced[i][len(generated)])
                if forced
                else int(z[:sampleable].argmax())
                for i, z in enumerate(step_logits)
            ]
        )
        rows = [
            Row(
                [generated[-1][i]],
                rows[i].cache_length + len(rows[i].tokens),
                rows[i].pages,
            )
            for i in range(len(prompts))
        ]
        step_logits = list(runner.step(caches, rows))
    return (
        [[g[i] for g in generated] for i in range(len(prompts))],
        [[z[i] for z in logits] for i in range(len(prompts))],
    )
