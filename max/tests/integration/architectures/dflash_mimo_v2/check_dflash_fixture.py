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
"""Checks the MAX MiMo-V2 DFlash drafter against vLLM's recorded drafts.

The fixture holds, for every speculative step vLLM took on 22 prompts, the
target taps the drafter read and the draft logits and tokens it produced.
This replays each step through the MAX drafter on one GPU and scores it
against vLLM, twice:

* one shot, each step a request in a ragged batch whose context is written
  from the position-indexed taps, pages below the window left on the null
  page;
* incrementally, one persistent cache per prompt written a step's committed
  rows at a time, as a serving engine would.

It also scores contexts of 1 to 8 positions, which vLLM never drafted from,
against the fixture's own torch reimplementation (``scripts/replay.py``, F32),
and reads the context K/V back out of the MAX cache to compare it with that
reference's. vLLM recorded ``fc(taps)`` but not the context K/V, so the
writer's first half is scored against vLLM and its K/V against the reference.

Needs the fixture and the checkpoint on disk, so it runs by hand, never in CI.
The reference runs on the CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np
import torch
from max import tree
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import load_weights
from max.nn.comm import Signals
from max.nn.embedding import Embedding
from max.nn.kv_cache import (
    PACKED_PAGE_STRIDE,
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    padded_lut_cols,
)
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.pipelines.architectures.dflash_mimo_v2 import (
    DFlashContextWriter,
    DFlashMiMoV2,
    DFlashMiMoV2Config,
    convert_safetensor_state_dict,
    load_mask_embedding,
)
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING_FILE,
)
from safetensors.torch import load_file

TAGS = [
    "p01_chat_nothink",  # spellchecker:disable-line
    "p02_chat_think",
    "p02_chat_think_long",
    "p03_tool_call",
    "p04_tool_result",
    "p05_code",
    "p06_chinese",
    "p07_long_code_9k",
    "p07_long_code_9k_prefixhit",
    "p08_long_needle_16k",
    *(f"b0{i}_gsm8k" for i in range(8)),
    *(f"b{i:02d}_humaneval" for i in range(8, 12)),
]
# The prefix-hit run recomputed only its tail; vLLM served the rest from the
# cache the second cold run filled.
FILL_FROM = {"p07_long_code_9k_prefixhit": "p07_long_code_9k_run2"}
# Token ids from here on are padding rows with no tokenizer entry.
SAMPLEABLE_VOCAB = 151_675
PAGE_SIZE = 128

# The bounds a BF16 drafter must meet against vLLM's drafts.
MIN_TOKEN_AGREEMENT = 0.99
MAX_KL_MEAN = 1e-3
MAX_KL_P99 = 1e-2
# A disagreement where vLLM's top-2 logits are further apart than this is a
# bug, not rounding.
MAX_DISAGREEMENT_MARGIN = 0.5
# About 2.5 standard errors of an agreement near 0.99 over ~4,000 drafts.
FLOOR_SLACK = 0.005


def _to_device_bf16(t: torch.Tensor, device: Device) -> Buffer:
    bits = t.contiguous().to(torch.bfloat16).view(torch.int16).numpy()
    return (
        Buffer.from_numpy(bits.view(np.uint16)).view(DType.bfloat16).to(device)
    )


def _to_torch(buffer: Buffer) -> torch.Tensor:
    host = buffer.to(CPU())
    if host.dtype == DType.bfloat16:
        bits = np.from_dlpack(host.view(DType.uint16)).copy()
        return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.from_dlpack(host).copy())


class _Target(Module):
    """The three target weights the drafter reads."""

    def __init__(self, vocab_size: int, hidden: int, device: DeviceRef) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            vocab_size, hidden, DType.bfloat16, device
        )
        self.lm_head = Linear(hidden, vocab_size, DType.bfloat16, device)
        self.norm = RMSNorm(
            hidden, DType.bfloat16, 1e-6, multiply_before_cast=False
        )

    def __call__(self, hidden: TensorValue) -> TensorValue:
        return self.lm_head(hidden)


@dataclass
class _Cache:
    """One paged drafter cache, a pool per device, with pages assigned by hand."""

    params: MHAKVCacheParams
    devices: Sequence[Device]
    num_pages: int
    kv_blocks: list[Buffer] = field(init=False)

    def __post_init__(self) -> None:
        # The page past the pool is the null page unassigned columns name.
        self.kv_blocks = [
            Buffer.zeros(
                [self.num_pages + 1, *self.params.shape_per_block],
                self.params.dtype,
                device,
            )
            for device in self.devices
        ]

    def inputs(
        self,
        pages: Sequence[Sequence[int | None]],
        cache_lengths: Sequence[int],
        prompt_lengths: Sequence[int],
    ) -> list[KVCacheInputsPerDevice[Buffer, Buffer]]:
        """Each device's cache inputs; ``None`` pages map to the null page."""
        seq_lens = [
            c + p for c, p in zip(cache_lengths, prompt_lengths, strict=True)
        ]
        cols = padded_lut_cols(max(len(p) for p in pages))
        table = np.full((len(pages), cols), self.num_pages, np.uint32)
        for row, request in enumerate(pages):
            for col, page in enumerate(request):
                if page is not None:
                    table[row, col] = page
        key = self.params.resolve_attn_key(
            len(pages), max(prompt_lengths), max(seq_lens)
        )
        return [
            KVCacheInputsPerDevice(
                kv_blocks=blocks,
                cache_lengths=Buffer.from_numpy(
                    np.array(cache_lengths, np.uint32)
                ).to(device),
                lookup_table=Buffer.from_numpy(table).to(device),
                max_prompt_length=Buffer.from_numpy(
                    np.array([max(prompt_lengths)], np.uint32)
                ),
                max_cache_length=Buffer.from_numpy(
                    np.array([max(seq_lens)], np.uint32)
                ),
                page_stride=Buffer.from_numpy(
                    np.array([PACKED_PAGE_STRIDE], np.int64)
                ),
                attention_dispatch_metadata=key.pack_into_buffer(
                    device, max(seq_lens)
                ),
            )
            for blocks, device in zip(self.kv_blocks, self.devices, strict=True)
        ]

    def read(
        self, pages: Sequence[int | None], positions: range
    ) -> torch.Tensor:
        """Returns ``[2, layers, len(positions), heads, head_dim]`` K and V,
        every device's heads side by side."""
        per_device = []
        for kv_blocks in self.kv_blocks:
            blocks = _to_torch(kv_blocks)
            rows = []
            for p in positions:
                page = pages[p // PAGE_SIZE]
                assert page is not None
                rows.append(blocks[page, :, :, p % PAGE_SIZE])
            per_device.append(torch.stack(rows, dim=2))
        return torch.cat(per_device, dim=-2)


class _Drafter:
    """The MAX drafter compiled into a context-write and a block graph."""

    def __init__(self, args: argparse.Namespace) -> None:
        dflash = Path(args.checkpoint) / "dflash"
        self.devices: list[Device] = [
            Accelerator(i) for i in range(args.devices)
        ]
        self.device = self.devices[0]
        self.refs = [DeviceRef.from_device(d) for d in self.devices]
        device_ref = self.refs[0]
        self.kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=5,
            page_size=PAGE_SIZE,
            devices=self.refs,
        )
        self.config = DFlashMiMoV2Config.from_dflash_config(
            json.loads((dflash / "config.json").read_text()),
            devices=self.refs,
            kv_params=self.kv_params,
            max_seq_len=args.max_seq_len,
        )
        # The tensor-parallel drafter's allreduces need them; one device
        # declares none.
        self.signal_types = (
            Signals(devices=self.refs).input_types() if args.devices > 1 else []
        )
        self.signals = (
            Signals.allocate(self.devices) if args.devices > 1 else []
        )
        drafter_weights = load_weights([dflash / DRAFT_WEIGHTS_FILE])
        state = convert_safetensor_state_dict(
            dict(drafter_weights.items()),
            self.config,
            load_mask_embedding(dflash / MASK_EMBEDDING_FILE, self.config),
        )
        self.drafter = DFlashMiMoV2(self.config)
        self.drafter.load_state_dict(state, strict=True)
        # The context is written by the writer alone, as the base graph with
        # the writer attached would; a module instance serves one graph.
        self.writer = DFlashContextWriter(self.config)
        self.writer.load_state_dict(
            {name: state[name] for name in self.writer.raw_state_dict()},
            strict=True,
        )

        target_weights = load_weights(
            [Path(args.checkpoint) / "model_pp0_ep0_shard0.safetensors"]
        )
        self.target = _Target(
            args.vocab_size, self.config.hidden_size, device_ref
        )
        self.target.load_state_dict(
            {
                "embed_tokens.weight": target_weights.model.embed_tokens.weight.data(),
                "lm_head.weight": target_weights.lm_head.weight.data(),
                "norm.weight": target_weights.model.norm.weight.data(),
            },
            strict=True,
        )
        # The target's final norm and the drafter's share the name
        # ``norm.weight``; unprefixed, one silently loads as the other.
        target_state = self.target.state_dict()
        for name, weight in self.target.raw_state_dict().items():
            weight.name = f"target.{name}"
        registry = {
            **self.drafter.state_dict(),
            **{f"target.{k}": v for k, v in target_state.items()},
        }
        assert len(registry) == len(self.drafter.state_dict()) + len(
            target_state
        )
        session = InferenceSession(devices=self.devices)
        self.write_model = session.load(
            self._write_graph(), weights_registry=registry
        )
        self.block_model = session.load(
            self._block_graph(), weights_registry=registry
        )

    def _write_graph(self) -> Graph:
        hidden = self.config.hidden_size
        num_taps = len(self.config.target_layer_ids)
        with Graph(
            "dflash_mimo_v2_write",
            input_types=[
                TensorType(
                    DType.bfloat16, ["rows", num_taps, hidden], self.refs[0]
                ),
                TensorType(DType.uint32, ["offsets"], self.refs[0]),
                *self.signal_types,
                *self.kv_params.flattened_kv_inputs(),
            ],
        ) as graph:
            taps, offsets, *rest = graph.inputs
            signals = [v.buffer for v in rest[: len(self.signal_types)]]
            kv = list(
                self.kv_params.unflatten_kv_inputs(
                    iter(rest[len(self.signal_types) :])
                )
            )
            taps_list = [taps.tensor[:, i, :] for i in range(num_taps)]
            fc = self.writer.combine_taps(
                [[t.to(r) for t in taps_list] for r in self.refs],
                signals or None,
            )
            self.writer.write(fc, [offsets.tensor.to(r) for r in self.refs], kv)
            graph.output(fc[0])
        return graph

    def _block_graph(self) -> Graph:
        hidden = self.config.hidden_size
        block = self.config.block_size
        with Graph(
            "dflash_mimo_v2_block",
            input_types=[
                TensorType(DType.int64, ["batch"], self.refs[0]),
                TensorType(DType.uint32, ["offsets"], self.refs[0]),
                *self.signal_types,
                *self.kv_params.flattened_kv_inputs(),
            ],
        ) as graph:
            anchors, offsets, *rest = graph.inputs
            signals = [v.buffer for v in rest[: len(self.signal_types)]]
            kv = list(
                self.kv_params.unflatten_kv_inputs(
                    iter(rest[len(self.signal_types) :])
                )
            )
            anchor_embeds = self.target.embed_tokens(anchors.tensor)
            rows = self.drafter.block_embeddings(
                [anchor_embeds.to(r) for r in self.refs]
            )
            h = self.drafter.forward_block(
                rows,
                kv,
                [offsets.tensor.to(r) for r in self.refs],
                signals or None,
            )[0]
            drafts = h.reshape((-1, block, hidden))[:, 1:, :].reshape(
                (-1, hidden)
            )
            graph.output(drafts, self.target(drafts))
        return graph

    def write(
        self,
        cache: _Cache,
        pages: Sequence[Sequence[int | None]],
        starts: Sequence[int],
        taps: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Writes each request's taps as context from ``starts``; returns fc."""
        lengths = [int(t.shape[0]) for t in taps]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
        (fc,) = self.write_model.execute(
            _to_device_bf16(torch.cat(list(taps)), self.device),
            Buffer.from_numpy(offsets).to(self.device),
            *self.signals,
            *tree.leaves(cache.inputs(pages, starts, lengths)),
        )
        assert isinstance(fc, Buffer)
        return _to_torch(fc)

    def block(
        self,
        cache: _Cache,
        pages: Sequence[Sequence[int | None]],
        anchor_positions: Sequence[int],
        anchor_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drafts one block per request; returns ``[B, 7, ...]`` hidden, logits."""
        batch = len(anchor_ids)
        block = self.config.block_size
        hidden, logits = self.block_model.execute(
            Buffer.from_numpy(np.array(anchor_ids, np.int64)).to(self.device),
            Buffer.from_numpy(
                (np.arange(batch + 1) * block).astype(np.uint32)
            ).to(self.device),
            *self.signals,
            *tree.leaves(
                cache.inputs(pages, anchor_positions, [block] * batch)
            ),
        )
        assert isinstance(hidden, Buffer) and isinstance(logits, Buffer)
        return (
            _to_torch(hidden).view(batch, block - 1, -1),
            _to_torch(logits).view(batch, block - 1, -1),
        )


@dataclass
class _Score:
    """Draft agreement with a reference over many draft positions."""

    agree: int = 0
    positions: int = 0
    first_agree: int = 0
    steps: int = 0
    kl: list[float] = field(default_factory=list)
    max_abs: list[float] = field(default_factory=list)
    cos_hidden: list[float] = field(default_factory=list)
    disagreement_margins: list[float] = field(default_factory=list)
    per_tag: dict[str, list[int]] = field(default_factory=dict)

    def add(
        self,
        tag: str,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        ref_tokens: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        ref_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scores one step's ``[7, vocab]`` logits; returns the drafts."""
        logits, ref_logits = logits.float(), ref_logits.float()
        tokens = logits[:, :SAMPLEABLE_VOCAB].argmax(-1)
        if ref_tokens is None:
            ref_tokens = ref_logits[:, :SAMPLEABLE_VOCAB].argmax(-1)
        ref_logp = torch.log_softmax(ref_logits, -1)
        logp = torch.log_softmax(logits, -1)
        self.kl += (ref_logp.exp() * (ref_logp - logp)).sum(-1).tolist()
        self.max_abs += (logits - ref_logits).abs().amax(-1).tolist()
        top2 = ref_logits[:, :SAMPLEABLE_VOCAB].topk(2, -1).values
        margin = top2[:, 0] - top2[:, 1]
        agree = tokens == ref_tokens
        self.disagreement_margins += margin[~agree].tolist()
        self.agree += int(agree.sum())
        self.positions += int(agree.numel())
        self.first_agree += int(agree[0])
        self.steps += 1
        tally = self.per_tag.setdefault(tag, [0, 0])
        tally[0] += int(agree.sum())
        tally[1] += int(agree.numel())
        if hidden is not None and ref_hidden is not None:
            self.cos_hidden += torch.nn.functional.cosine_similarity(
                hidden.float(), ref_hidden.float(), dim=-1
            ).tolist()
        return tokens

    def summary(self) -> dict[str, object]:
        kl = torch.tensor(self.kl)
        out: dict[str, object] = {
            "draft_positions": self.positions,
            "steps": self.steps,
            "token_agreement": self.token_agreement,
            "first_draft_agreement": self.first_draft_agreement,
            "kl_mean": float(kl.mean()),
            "kl_p99": float(kl.quantile(0.99)),
            "kl_max": float(kl.max()),
            "logit_max_abs_mean": float(torch.tensor(self.max_abs).mean()),
            "disagreement_margins": sorted(self.disagreement_margins),
            "per_tag_agreement": {
                t: a / n for t, (a, n) in sorted(self.per_tag.items())
            },
        }
        if self.cos_hidden:
            out["hidden_cos_min"] = min(self.cos_hidden)
        return out

    @property
    def token_agreement(self) -> float:
        return self.agree / max(1, self.positions)

    @property
    def first_draft_agreement(self) -> float:
        return self.first_agree / max(1, self.steps)

    def failures(self, name: str, floor: _Score | None = None) -> list[str]:
        """Checks the bounds; ``floor`` lowers the agreement bounds to its own.

        A BF16 drafter cannot agree with an F32 one more often than the
        reference's BF16 mode does, so against the reference the agreement
        bounds become that mode's agreement less ``FLOOR_SLACK``.
        """
        min_agreement = min_first = MIN_TOKEN_AGREEMENT
        if floor is not None:
            min_agreement = min(
                min_agreement, floor.token_agreement - FLOOR_SLACK
            )
            min_first = min(
                min_first, floor.first_draft_agreement - FLOOR_SLACK
            )
        kl = torch.tensor(self.kl)
        # (what, value, bound, whether the bound is a minimum)
        checks = [
            ("token agreement", self.token_agreement, min_agreement, True),
            (
                "first-draft agreement",
                self.first_draft_agreement,
                min_first,
                True,
            ),
            ("mean KL", float(kl.mean()), MAX_KL_MEAN, False),
            ("KL p99", float(kl.quantile(0.99)), MAX_KL_P99, False),
            (
                "largest disagreeing margin",
                max(self.disagreement_margins, default=0.0),
                MAX_DISAGREEMENT_MARGIN,
                False,
            ),
        ]
        return [
            f"{name}: {what} {value:.4g} {'<' if minimum else '>'} {bound:.4g}"
            for what, value, bound, minimum in checks
            if (value < bound if minimum else value > bound)
        ]


@dataclass
class _Recording:
    tag: str
    taps: torch.Tensor
    valid: torch.Tensor
    fc_out: torch.Tensor
    tokens: torch.Tensor
    steps: dict[str, torch.Tensor]

    @classmethod
    def load(cls, fixture: Path, tag: str) -> _Recording:
        tensors = load_file(fixture / tag / "taps.safetensors")
        rec = cls(
            tag,
            tensors["taps"],
            tensors["valid"].bool(),
            tensors["fc_out"],
            tensors["token"],
            load_file(fixture / tag / "steps.safetensors"),
        )
        if tag in FILL_FROM:
            src = load_file(fixture / FILL_FROM[tag] / "taps.safetensors")
            n = min(src["valid"].numel(), rec.valid.numel())
            fill = ~rec.valid[:n] & src["valid"][:n].bool()
            rec.taps[:n][fill] = src["taps"][:n][fill]
            rec.fc_out[:n][fill] = src["fc_out"][:n][fill]
            rec.tokens[:n][fill] = src["token"][:n][fill]
            rec.valid[:n][fill] = True
        return rec

    def decode_steps(self) -> list[int]:
        return torch.nonzero(self.steps["is_decode"]).flatten().tolist()


def _window_pages(
    lo: int, end: int, first_free: int
) -> tuple[list[int | None], int]:
    """Real pages for ``[lo, end)``, null pages below; returns the next free."""
    pages: list[int | None] = [None] * (lo // PAGE_SIZE)
    for _ in range(lo // PAGE_SIZE, -(-end // PAGE_SIZE)):
        pages.append(first_free)
        first_free += 1
    return pages, first_free


def run_one_shot(
    drafter: _Drafter,
    rec: _Recording,
    score: _Score,
    fc_rel: list[float],
    batch: int,
    kv_check: list[tuple[_Recording, int, torch.Tensor]] | None = None,
) -> dict[int, torch.Tensor]:
    """Scores every decode step, each a request in a ragged batch."""
    window = drafter.config.sliding_window - 1
    block = drafter.config.block_size
    drafts = {}
    steps = rec.decode_steps()
    pages_per_request = -(-(window + block) // PAGE_SIZE) + 1
    cache = _Cache(
        drafter.kv_params, drafter.devices, batch * pages_per_request
    )
    for start in range(0, len(steps), batch):
        chunk = steps[start : start + batch]
        anchors = [int(rec.steps["anchor_pos"][s]) for s in chunk]
        los = [max(0, p - window) for p in anchors]
        spans = list(zip(los, anchors, strict=True))
        assert all(bool(rec.valid[lo:p].all()) for lo, p in spans)
        pages, free = [], 0
        for lo, p in spans:
            request_pages, free = _window_pages(lo, p + block, free)
            pages.append(request_pages)
        fc = drafter.write(
            cache, pages, los, [rec.taps[lo:p] for lo, p in spans]
        )
        ref_fc = torch.cat([rec.fc_out[lo:p] for lo, p in spans]).float()
        fc_rel.append(float((fc.float() - ref_fc).norm() / ref_fc.norm()))
        if kv_check is not None and start == 0:
            kv_check.append(
                (rec, chunk[0], cache.read(pages[0], range(los[0], anchors[0])))
            )
        ids = [int(rec.steps["block_input_ids"][s, 0]) for s in chunk]
        hidden, logits = drafter.block(cache, pages, anchors, ids)
        for i, s in enumerate(chunk):
            score.add(
                rec.tag,
                logits[i],
                rec.steps["draft_logits"][s],
                rec.steps["draft_tokens"][s],
                hidden[i],
                rec.steps["draft_hidden"][s],
            )
            drafts[s] = logits[i].float()
    return drafts


def run_incremental(
    drafter: _Drafter, rec: _Recording, score: _Score
) -> dict[int, torch.Tensor]:
    """Scores every decode step on one persistent cache, written step by step."""
    block = drafter.config.block_size
    length = int(rec.valid.numel()) + block
    cache = _Cache(drafter.kv_params, drafter.devices, -(-length // PAGE_SIZE))
    pages = [list(range(cache.num_pages))]
    first = int(rec.steps["ctx_first_pos"][0])
    assert bool(rec.valid[:first].all())
    if first:
        # The prefix-hit run starts on cached context.
        drafter.write(cache, pages, [0], [rec.taps[:first]])
    drafts = {}
    for s in range(int(rec.steps["anchor_pos"].numel())):
        lo = int(rec.steps["ctx_first_pos"][s])
        hi = lo + int(rec.steps["ctx_rows_written"][s])
        drafter.write(cache, pages, [lo], [rec.taps[lo:hi]])
        anchor = int(rec.steps["anchor_pos"][s])
        assert anchor == hi, f"{rec.tag} step {s}: anchor {anchor} != {hi}"
        if not bool(rec.steps["is_decode"][s]):
            continue
        _, logits = drafter.block(
            cache, pages, [anchor], [int(rec.steps["block_input_ids"][s, 0])]
        )
        score.add(
            rec.tag,
            logits[0],
            rec.steps["draft_logits"][s],
            rec.steps["draft_tokens"][s],
        )
        drafts[s] = logits[0].float()
    return drafts


class _Reference(Protocol):
    """``scripts/replay.py``'s ``Drafter``, the fixture's torch reimplementation."""

    def draft(
        self,
        taps: torch.Tensor,
        ctx_pos: torch.Tensor,
        anchor_id: int,
        P: int,
        abl: tuple[str, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def context_kv(
        self, taps: torch.Tensor, pos: torch.Tensor, abl: tuple[str, ...]
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]: ...


def _load_reference(fixture: Path, checkpoint: str) -> ModuleType:
    os.environ["MIMO_CKPT"] = checkpoint
    spec = importlib.util.spec_from_file_location(
        "dflash_replay", fixture / "scripts" / "replay.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_short_contexts(
    drafter: _Drafter,
    references: dict[str, _Reference],
    recordings: Sequence[_Recording],
    score: _Score,
    floor: _Score,
) -> None:
    """Scores contexts of 1 to 8 positions against the F32 reference.

    Each context is all its cache holds, at positions ``[0, n)`` with the
    anchor at ``n``. Real sequence starts are few, since the chat template
    makes many prompts start alike, so the taps and anchor of spans from
    three more places in each sequence are drafted at those positions too.
    Each distinct span is scored once.

    ``floor`` scores the reference's own BF16 mode the same way: how far any
    BF16 drafter lands from F32 on these contexts, which vLLM never drafted.
    """
    block = drafter.config.block_size
    seen: set[tuple[int, ...]] = set()
    for rec in recordings:
        last = int(rec.valid.nonzero().max())
        cases = []
        for q in [0, *np.linspace(1, last - 8, 3).astype(int).tolist()]:
            for n in range(1, 9):
                key = (n, *rec.tokens[q : q + n + 1].tolist())
                if key not in seen and bool(rec.valid[q : q + n].all()):
                    seen.add(key)
                    cases.append((q, n))
        assert all(n + block <= PAGE_SIZE for _, n in cases)
        pages = [[i] for i in range(len(cases))]
        cache = _Cache(drafter.kv_params, drafter.devices, len(cases))
        taps = [rec.taps[q : q + n] for q, n in cases]
        drafter.write(cache, pages, [0] * len(cases), taps)
        anchors = [int(rec.tokens[q + n]) for q, n in cases]
        lengths = [n for _, n in cases]
        _, logits = drafter.block(cache, pages, lengths, anchors)
        for i, n in enumerate(lengths):
            ref_logits, ref_bf16_logits = (
                references[mode].draft(taps[i], torch.arange(n), anchors[i], n)[
                    2
                ]
                for mode in ("f32", "bf16")
            )
            score.add(rec.tag, logits[i], ref_logits)
            floor.add(rec.tag, ref_bf16_logits, ref_logits)


def check_context_kv(
    references: dict[str, _Reference],
    checks: Sequence[tuple[_Recording, int, torch.Tensor]],
) -> dict[str, float]:
    """Compares the MAX cache's context K/V with the reference's.

    Returns the worst relative error of MAX against the F32 reference, and,
    as the scale that error should be read at, of the reference's own BF16
    mode against its F32 mode.
    """
    worst = {"max_k": 0.0, "max_v": 0.0, "ref_bf16_k": 0.0, "ref_bf16_v": 0.0}

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).norm() / b.float().norm())

    for rec, step, cached in checks:
        anchor = int(rec.steps["anchor_pos"][step])
        lo = max(0, anchor - 1023)
        pos = torch.arange(lo, anchor)
        _, f32 = references["f32"].context_kv(rec.taps[pos], pos, ())
        _, bf16 = references["bf16"].context_kv(rec.taps[pos], pos, ())
        for layer, ((k, v), (kb, vb)) in enumerate(zip(f32, bf16, strict=True)):
            worst["max_k"] = max(worst["max_k"], rel(cached[0, layer], k))
            worst["max_v"] = max(worst["max_v"], rel(cached[1, layer], v))
            worst["ref_bf16_k"] = max(worst["ref_bf16_k"], rel(kb, k))
            worst["ref_bf16_v"] = max(worst["ref_bf16_v"], rel(vb, v))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="checkpoint root, holding dflash/"
    )
    parser.add_argument(
        "--fixture", required=True, help="the DFlash fixture directory"
    )
    parser.add_argument("--tags", default=",".join(TAGS))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="tensor-parallel width of the drafter, row-parallel fc included",
    )
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--vocab-size", type=int, default=152576)
    parser.add_argument("--out", help="write the results as JSON here")
    args = parser.parse_args()

    torch.set_num_threads(min(32, os.cpu_count() or 1))
    fixture = Path(args.fixture).expanduser()
    drafter = _Drafter(args)
    replay = _load_reference(fixture, str(Path(args.checkpoint).expanduser()))
    references: dict[str, _Reference] = {
        mode: replay.Drafter("cpu", mode) for mode in ("f32", "bf16")
    }

    one_shot, incremental = _Score(), _Score()
    short, short_floor = _Score(), _Score()
    fc_rel: list[float] = []
    kv_checks: list[tuple[_Recording, int, torch.Tensor]] = []
    paths_max_abs, paths_agree, paths_n = 0.0, 0, 0
    recordings = []
    for tag in args.tags.split(","):
        rec = _Recording.load(fixture, tag)
        recordings.append(rec)
        a = run_one_shot(drafter, rec, one_shot, fc_rel, args.batch, kv_checks)
        b = run_incremental(drafter, rec, incremental)
        for s, logits in a.items():
            paths_max_abs = max(
                paths_max_abs, float((logits - b[s]).abs().max())
            )
            paths_agree += int(
                (
                    logits[:, :SAMPLEABLE_VOCAB].argmax(-1)
                    == b[s][:, :SAMPLEABLE_VOCAB].argmax(-1)
                ).sum()
            )
            paths_n += logits.shape[0]
        print(
            f"{tag}: one-shot {one_shot.per_tag[tag][0]}/{one_shot.per_tag[tag][1]},"
            f" incremental {incremental.per_tag[tag][0]}/{incremental.per_tag[tag][1]}",
            flush=True,
        )
    run_short_contexts(drafter, references, recordings, short, short_floor)
    kv = check_context_kv(references, kv_checks)

    results = {
        "one_shot_vs_vllm": one_shot.summary(),
        "incremental_vs_vllm": incremental.summary(),
        "one_shot_vs_incremental": {
            "token_agreement": paths_agree / max(1, paths_n),
            "logit_max_abs": paths_max_abs,
        },
        "short_contexts_vs_reference_f32": short.summary(),
        "short_contexts_reference_bf16_vs_f32": short_floor.summary(),
        "fc_rel_err_vs_vllm_max": max(fc_rel),
        "context_kv_rel_err": kv,
    }
    failures = (
        one_shot.failures("one shot vs vLLM")
        + incremental.failures("incremental vs vLLM")
        + short.failures("short contexts vs reference", floor=short_floor)
    )
    # K/V may sit no further from the F32 reference than a few times the
    # reference's own BF16 rounding.
    for which in ("k", "v"):
        bound = 4 * kv[f"ref_bf16_{which}"]
        if kv[f"max_{which}"] > bound:
            failures.append(
                f"context {which.upper()} rel err {kv[f'max_{which}']:.3g} > {bound:.3g}"
            )
    results["failures"] = failures
    print(json.dumps({k: v for k, v in results.items()}, indent=1, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
