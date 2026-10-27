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
"""A tiny random-weight MiMo-V2 target beside a tiny DFlash drafter, fused as
served.

The fused graph comes from ``build_spec_graph``, the
code a served pairing runs, so the graph under test is the production one. The
caches are paged pools whose pages the caller assigns, and the runners drive
the compiled graphs by hand on one device.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast
from unittest import mock

import ml_dtypes
import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.weights import WeightData
from max.nn.comm import Signals
from max.nn.kv_cache import (
    PACKED_PAGE_STRIDE,
    KVCacheInputsPerDevice,
    KVCacheParams,
    MultiKVCacheParams,
    padded_lut_cols,
)
from max.nn.layer import Module
from max.nn.sampling import AcceptanceSampler
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.dflash_mimo_v2 import (
    DFlashMiMoV2,
    DFlashMiMoV2DraftConfig,
    DrafterExport,
)
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    MASK_EMBEDDING,
)
from max.pipelines.architectures.mimo_v2.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2.model_config import (
    FULL,
    SLIDING,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.pipeline_variants import spec_decode_pipeline_model
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    build_spec_graph,
)
from max.pipelines.lib.pipeline_variants.structured_output_overlap import (
    StructuredOutputOverlapState,
)
from max.pipelines.speculative import SpeculativeConfig
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.weights._fp8 import e4m3fn_lut
from tiny_mimo_dflash import (
    PAGE_SIZE,
    Model,
    tiny_dflash_config,
    tiny_target_hf,
    weight_data,
)

TARGET_ARCH = "MiMoV2ForCausalLM"
DRAFT_ARCH = "DFlashMiMoV2DraftModel"
GRAPH_NAME = "unified_dflash_mimo_v2"

SAMPLEABLE = 1000
"""The tiny model's tokenizer size: ids from here to its vocab's 1,024 stand
in for the ``lm_head`` padding rows."""

F32 = npt.NDArray[np.float32]


def bf16_bits(x: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Rounds to the nearest BFloat16 and returns its bits."""
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)


def to_f32(buffer: Buffer) -> F32:
    """Returns a host copy of ``buffer`` as float32."""
    host = buffer.to(CPU())
    if host.dtype == DType.bfloat16:
        return (
            np.from_dlpack(host.view(DType.uint16))
            .view(ml_dtypes.bfloat16)
            .astype(np.float32)
        )
    return np.from_dlpack(host).astype(np.float32)


def random_state(
    module: Module, rng: np.random.Generator
) -> dict[str, WeightData]:
    """Returns random weights for every weight ``module`` declares, sized to
    keep activations O(1) through a pre-norm residual stack."""
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
            elif "sink" in name or "embed" in name:
                values = rng.standard_normal(shape)
            elif len(shape) == 2:
                values = rng.standard_normal(shape) / math.sqrt(shape[1])
            else:
                values = 0.1 * rng.standard_normal(shape)
            array = bf16_bits(values)
        state[name] = weight_data(name, dtype, array)
    return state


def tiny_model(seed: int = 0, *, draft_kv_heads: int | None = None) -> Model:
    """Returns a random-weight target and drafter in the serving layouts.

    Args:
        seed: The seed of the random weights.
        draft_kv_heads: The drafter's KV head count, when it shards wider
            than the tiny drafter's two heads allow.
    """
    rng = np.random.default_rng(seed)
    hf = tiny_target_hf()
    dflash = tiny_dflash_config(list(range(hf.num_hidden_layers)))
    if draft_kv_heads is not None:
        dflash["num_key_value_heads"] = draft_kv_heads
    model = Model(
        hf=hf,
        dflash=dflash,
        target_state={},
        draft_state={},
        mask_embedding=weight_data(
            "mask_embedding", DType.bfloat16, np.zeros(1, np.uint16)
        ),
        max_seq_len=4096,
    )
    one = [DeviceRef.GPU()]
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


def sampleable_bitmask(
    batch: int, positions: int, vocab: int, sampleable: int
) -> npt.NDArray[np.int32]:
    """Returns the packed bitmask that allows exactly the ids below
    ``sampleable``.

    Bit ``t % 32`` of word ``t // 32`` is token ``t``, as the kernel reads it.
    """
    words = np.zeros(-(-vocab // 32), np.uint32)
    words[: sampleable // 32] = 0xFFFFFFFF
    if sampleable % 32:
        words[sampleable // 32] = (1 << (sampleable % 32)) - 1
    return np.ascontiguousarray(
        np.broadcast_to(words.view(np.int32), (batch, positions, words.size))
    )


class RecordingSampler(AcceptanceSampler):
    """Forwards to an acceptance sampler and keeps the logits it verified
    against, optionally accepting one draft past the first rejection.

    With ``accept_one_extra`` the committed token is the target's own
    prediction after the extra draft, so the output stays self-consistent
    except at that draft, which the target did not choose.
    """

    def __init__(
        self, sampler: AcceptanceSampler, *, accept_one_extra: bool
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


class VerifyLogitsDriver(BlockDriver[Any, Any]):
    """The block driver with the logits acceptance verified against as a
    fourth output: ``[batch * (K + 1), vocab]``, or ``[batch, vocab]`` on a
    prefill."""

    def __init__(
        self, *args: Any, accept_one_extra: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recording = RecordingSampler(
            self.acceptance_sampler, accept_one_extra=accept_one_extra
        )
        self.acceptance_sampler = self.recording

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Pass-through return: BlockDriver's __call__ is generic over the
        # value type, which an override cannot restate from *args; the
        # appended verify logits and the base outputs are documented above
        # and splatted into graph.output by the caller.
        outputs = super().__call__(*args, **kwargs)
        assert self.recording.target_logits is not None
        return (*outputs, self.recording.target_logits)


class FusedGraph(NamedTuple):
    """A fused graph, its weights and the KV groups it declares."""

    graph: Graph
    weights_registry: dict[str, Any]
    groups: dict[str, KVCacheParams]
    block: int


def fused_graph(
    model: Model,
    devices: Sequence[DeviceRef],
    k: int,
    *,
    greedy: bool = True,
    accept_one_extra: bool = False,
    verify_logits: bool = False,
    in_thinking_phase: bool = False,
    fused_sliding_qkv: bool = False,
    fused_full_qkv: bool = False,
) -> FusedGraph:
    """Builds the fused graph a served MiMo-V2 DFlash pairing compiles.

    Args:
        pairing: The target and drafter.
        devices: The devices to shard over.
        k: How many proposals a step verifies.
        greedy: Whether acceptance is greedy rather than rejection sampling
            under each row's own parameters.
        accept_one_extra: Whether to build in a defect that accepts one draft
            past the first rejection, to prove a check can fail.
        verify_logits: Whether to append the verified logits as an output.
        in_thinking_phase: Whether the pairing serves a reasoning model.
        fused_sliding_qkv: Whether the target's sliding layers prepare QKV
            with the fused op.
        fused_full_qkv: Whether the target's full layers prepare QKV with
            the fused op.
    """
    devices = list(devices)
    speculative = SpeculativeConfig(
        speculative_method="dflash",
        num_speculative_tokens=k,
        use_greedy_acceptance=greedy,
    )
    target_cfg = model.target(devices, speculative=True)
    target_cfg.return_logits = ReturnLogits.VARIABLE
    target_cfg.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
    target_cfg.target_layer_ids = list(
        model.dflash["dflash_config"]["target_layer_ids"]
    )
    target_cfg.fused_sliding_qkv = fused_sliding_qkv
    target_cfg.fused_full_qkv = fused_full_qkv
    target = MiMoV2(target_cfg)
    target.load_state_dict(model.target_state, weight_alignment=1, strict=True)
    module_cfg = model.drafter(devices, speculative=True)
    draft = DFlashMiMoV2(module_cfg, mask_embedding=model.mask_embedding)
    draft.load_state_dict(model.draft_state, weight_alignment=1, strict=True)
    draft_weights = dict(draft.state_dict())
    # The drafter embeds and projects through its target's embedding and head.
    draft.embed_tokens = target.embed_tokens
    draft.lm_head = target.lm_head
    draft_cfg = DFlashMiMoV2DraftConfig(
        module=module_cfg,
        sampleable_vocab_size=SAMPLEABLE,
        drafter_export=DrafterExport(
            directory=Path(),
            target_layer_ids=list(module_cfg.target_layer_ids),
            num_speculative_tokens=k,
            spec_block_size=module_cfg.block_size,
        ),
    )
    assert isinstance(target_cfg.kv_params, MultiKVCacheParams)
    kv_tree = MultiKVCacheParams.from_params(
        {"target": target_cfg.kv_params, "draft": module_cfg.kv_params}
    )
    assert PIPELINE_REGISTRY.retrieve_architecture(TARGET_ARCH) is not None
    assert PIPELINE_REGISTRY.retrieve_architecture(DRAFT_ARCH) is not None
    pipeline_config = SimpleNamespace(
        models=SimpleNamespace(main_architecture_name=TARGET_ARCH),
        draft_model=SimpleNamespace(
            huggingface_config=SimpleNamespace(architectures=[DRAFT_ARCH])
        ),
        speculative=speculative,
        needs_bitmask_constraints=True,
        runtime=SimpleNamespace(
            reasoning_parser="mimo" if in_thinking_phase else None
        ),
        model=SimpleNamespace(
            data_parallel_degree=1,
            huggingface_config=model.hf,
            pipeline_stage=None,
        ),
    )
    driver = (
        partial(VerifyLogitsDriver, accept_one_extra=accept_one_extra)
        if verify_logits
        else BlockDriver
    )
    with mock.patch.object(spec_decode_pipeline_model, "BlockDriver", driver):
        spec = build_spec_graph(
            cast("PipelineConfig", pipeline_config),
            kv_tree,
            devices,
            # MiMo-V2 declares signal buffers whatever its device count.
            include_signal_buffers=True,
            target=target,
            target_config=target_cfg,
            draft=draft,
            draft_config=draft_cfg,
            hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            draft_weights=draft_weights,
        )
    graph, registry = spec.graph, spec.weights_registry
    groups: dict[str, KVCacheParams] = {}
    for name in (SLIDING, FULL):
        leaf = target_cfg.kv_params.children[name]
        assert isinstance(leaf, KVCacheParams)
        groups[name] = leaf
    groups["draft"] = module_cfg.kv_params
    return FusedGraph(graph, registry, groups, module_cfg.block_size)


class Pages:
    """Hands out page ids, never reusing one."""

    def __init__(self) -> None:
        self.next = 0

    def take(self, positions: int) -> list[int]:
        """Returns fresh pages covering ``positions``."""
        count = -(-positions // PAGE_SIZE)
        pages = list(range(self.next, self.next + count))
        self.next += count
        return pages


class PagedTree:
    """One page pool per KV group, on one device.

    The pools hold no speculative state, so a speculative graph and a base
    graph can share them; which dispatch buffers a step passes comes from the
    params each graph was built with.
    """

    def __init__(
        self,
        groups: Mapping[str, KVCacheParams],
        device: Device,
        num_pages: int,
    ) -> None:
        self.device = device
        self.num_pages = num_pages
        # The page past the pool is the null page every unassigned lookup
        # column names.
        self.blocks = {
            name: Buffer.zeros(
                [num_pages + 1, *params.shape_per_block],
                params.dtype,
                device,
            )
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
        """Returns the flattened KV inputs of every group, in ``groups``' order.

        Args:
            groups: The params the graph declared its KV inputs from.
            pages: Per request, its pages in position order.
            cache_lengths: Per request, the positions already cached.
            row_lengths: Per request, the rows this step writes.
            reach: Per group, how far past a request's rows the step reads,
                for a block written beyond them.
        """
        device = self.device
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
            flat += tree.leaves(
                KVCacheInputsPerDevice(
                    kv_blocks=self.blocks[name],
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


def _sampling_inputs(
    rows: Sequence[Sampling], generated: Sequence[int], device: Device
) -> tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer]:
    """Returns the sampling inputs: per-row seed, temperature, top-k and
    top-p, and the batch's largest top-k and smallest top-p on the host."""
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
        model: Model,
        k: int,
        device: Device,
        session: InferenceSession,
        *,
        greedy: bool = True,
        accept_one_extra: bool = False,
        target_state: Mapping[str, WeightData] | None = None,
        bitmask_allows_padding: bool = False,
        max_batch_size: int = 16,
    ) -> None:
        self.device = device
        self.k = k
        if target_state is not None:
            model = replace(model, target_state=dict(target_state))
        built = fused_graph(
            model,
            [DeviceRef.from_device(device)],
            k,
            greedy=greedy,
            accept_one_extra=accept_one_extra,
            verify_logits=True,
        )
        self.model = session.load(
            built.graph, weights_registry=built.weights_registry
        )
        self.groups = built.groups
        self.block = built.block
        self.signals = Signals.allocate([device])
        self.vocab = int(model.hf.vocab_size)
        # Allowing the padding rows leaves the graph's own mask alone keeping
        # them out.
        self.allowed = self.vocab if bitmask_allows_padding else SAMPLEABLE
        self.bitmask = StructuredOutputOverlapState(
            device, CPU(), max_batch_size, [1, k + 1], self.vocab
        )

    def step(self, caches: PagedTree, rows: Sequence[Row]) -> FusedStep:
        """Runs one step; every row carries K drafts, or every row none."""
        drafts = {len(r.drafts) for r in rows}
        assert drafts in ({0}, {self.k}), drafts
        steps = drafts.pop()
        device = self.device
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
                reach={"draft": self.block},
            ),
            Buffer.from_numpy(draft_tokens).to(device),
            *_sampling_inputs(
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


class BaseRunner:
    """The MiMo-V2 module on its own, returning every row's logits: plain
    one-token-at-a-time decoding to compare speculation against."""

    def __init__(
        self,
        model: Model,
        device: Device,
        session: InferenceSession,
    ) -> None:
        self.device = device
        ref = DeviceRef.from_device(device)
        config = model.target([ref], speculative=False)
        config.return_logits = ReturnLogits.ALL
        target_kv = config.kv_params
        assert isinstance(target_kv, MultiKVCacheParams)
        self.groups: dict[str, KVCacheParams] = {}
        for name in (SLIDING, FULL):
            leaf = target_kv.children[name]
            assert isinstance(leaf, KVCacheParams)
            self.groups[name] = leaf
        module = MiMoV2(config)
        module.load_state_dict(
            model.target_state, weight_alignment=1, strict=True
        )
        input_types = [
            TensorType(DType.int64, ["total_seq_len"], device=ref),
            TensorType(
                DType.int64, ["return_n_logits"], device=DeviceRef.CPU()
            ),
            TensorType(DType.uint32, ["input_row_offsets_len"], device=ref),
            *Signals(devices=[ref]).input_types(),
            *tree.leaves(target_kv.get_symbolic_inputs()),
        ]
        with Graph("mimo_v2", input_types=input_types) as graph:
            tokens, return_n_logits, offsets, signal, *kv = graph.inputs
            sliding, full = target_kv.unflatten_basic_kv_tree(iter(kv))
            graph.output(
                *module(
                    tokens=tokens.tensor,
                    signal_buffers=[signal.buffer],
                    sliding_kv_collections=sliding,
                    full_kv_collections=full,
                    return_n_logits=return_n_logits.tensor,
                    input_row_offsets=[offsets.tensor],
                )
            )
        self.model = session.load(
            graph, weights_registry=module.state_dict(auto_initialize=False)
        )
        self.signals = Signals.allocate([device])

    def step(self, caches: PagedTree, rows: Sequence[Row]) -> F32:
        """Runs one step; returns ``[rows, vocab]`` logits for every row."""
        tokens = np.concatenate([np.array(r.tokens, np.int64) for r in rows])
        offsets = np.cumsum([0] + [len(r.tokens) for r in rows]).astype(
            np.uint32
        )
        outputs = self.model.execute(
            Buffer.from_numpy(tokens).to(self.device),
            Buffer.from_numpy(np.array([1], np.int64)),
            Buffer.from_numpy(offsets).to(self.device),
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


def committed_token_violations(
    rows: Sequence[Row], step: FusedStep
) -> list[str]:
    """Returns where a step committed anything but the target's own greedy
    choice.

    Row ``j`` of a request's verify logits predicts the token after its
    ``j``-th verified position, so the first ``a`` drafts must each be that
    row's argmax, bitmask applied, the committed token row ``a``'s, and the
    next draft must not match, or it too would have been accepted.
    """
    problems = []
    for i, row in enumerate(rows):
        target = step.logits[i][:, :SAMPLEABLE].argmax(-1)
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
    """Speculative decoding's output, per request."""

    tokens: list[list[int]]
    """Committed output tokens, cut at the limit."""
    accepted: list[list[int]]
    """Per drafted verify step, the drafts it accepted."""
    violations: list[str]


def generate(
    runner: FusedRunner,
    caches: PagedTree,
    pages: Pages,
    prompts: Sequence[Sequence[int]],
    max_new: int,
    *,
    guide: Sequence[Sequence[int]] | None = None,
    rng: np.random.Generator | None = None,
    sampling: Sequence[Sampling] | None = None,
) -> Generation:
    """Runs speculative decoding through the fused graph, one batch; greedy
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
    result = Generation(
        tokens=[[] for _ in prompts],
        accepted=[[] for _ in prompts],
        violations=committed_token_violations(rows, out),
    )
    lengths = [len(p) for p in prompts]
    anchors = [int(t) for t in out.next_tokens]
    drafts = out.next_drafts.tolist()
    live = []
    for i, token in enumerate(anchors):
        result.tokens[i].append(token)
        if max_new > 1:
            live.append(i)
    while live:
        if guide is not None:
            assert rng is not None
            for i in live:
                done = len(result.tokens[i])
                proposal = list(guide[i][done : done + k])
                proposal += [0] * (k - len(proposal))
                if rng.random() < 0.5:
                    proposal[rng.integers(k)] = int(rng.integers(SAMPLEABLE))
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
        result.violations += committed_token_violations(step_rows, out)
        still = []
        for j, i in enumerate(live):
            a = int(out.num_accepted[j])
            new = step_rows[j].drafts[:a] + [int(out.next_tokens[j])]
            stopped = False
            for token in new:
                result.tokens[i].append(token)
                if len(result.tokens[i]) >= max_new:
                    stopped = True
                    break
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
    forced: Sequence[Sequence[int]] | None = None,
) -> tuple[list[list[int]], list[list[F32]]]:
    """Runs plain greedy decoding, one token a step, or teacher-forced on
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
                else int(z[:SAMPLEABLE].argmax())
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
