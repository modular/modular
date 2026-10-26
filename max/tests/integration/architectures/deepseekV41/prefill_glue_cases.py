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
"""Graphs of DeepSeek-V4.1's prefill glue, old path or new.

:func:`compressor_model` builds the compressor's projection on the
checkpoint's bf16 weights or on weights widened to float32, and
:func:`index_lists_model` builds a forward's 40 trunk index lists from
either of the functions ``CSA2Attention._index_list`` dispatches to, so a
test can compare the two paths on the same inputs and a bench can time them
at the model's shapes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import torch
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.pipelines.architectures.deepseekV41.layer_resolver import resolve
from max.pipelines.architectures.deepseekV41.layers.compressor import (
    DeepseekV41Compressor,
)
from max.pipelines.architectures.deepseekV41.layers.csa2 import (
    index_list,
    index_list_ops,
)
from max.pipelines.architectures.deepseekV41.layers.indexer import (
    key_positions,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)

HIDDEN = 5120
HEAD_DIM = 512
# DeepSeek-V4.1-Flash's RMSNorm epsilon.
RMS_EPS = 1e-20


def hidden(rows: int, seed: int) -> torch.Tensor:
    """Draws ``rows`` standard normal hidden states, sampled in bf16."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        rows, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16
    )


class _Compressor(Module):
    def __init__(self, ratio: int, weight_dtype: DType) -> None:
        super().__init__()
        self.compressor = DeepseekV41Compressor(
            ratio,
            RMS_EPS,
            hidden_size=HIDDEN,
            device=DeviceRef.GPU(),
            weight_dtype=weight_dtype,
        )

    def __call__(self) -> None:
        raise NotImplementedError("a weight holder; trace with its members")


def compressor_weights(ratio: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Returns bf16 ``wkv``/``wgate`` weights, as the checkpoint stores them."""
    gen = torch.Generator().manual_seed(seed)
    names = ["wkv"] if ratio == 1 else ["wkv", "wgate"]
    return {
        f"compressor.{name}.weight": (
            torch.randn(HEAD_DIM, HIDDEN, generator=gen) / math.sqrt(HIDDEN)
        ).to(torch.bfloat16)
        for name in names
    }


def compressor_norm_weight() -> torch.Tensor:
    """The pooled latent's RMSNorm weight, which ``project`` does not read."""
    return torch.ones(HEAD_DIM, dtype=torch.float32)


def compressor_model(
    session: InferenceSession,
    ratio: int,
    bf16: bool,
    weights: dict[str, torch.Tensor],
) -> Model:
    """The compressor's ``project``: ``[s, d]`` bf16 -> ``[s, 512 * ratio]``.

    ``bf16`` keeps the checkpoint's bf16 weights; otherwise they are widened
    to float32 as the weight adapter does with the switch off.
    """
    dtype = DType.bfloat16 if bf16 else DType.float32
    module = _Compressor(ratio, dtype)
    module.load_state_dict(
        {
            "compressor.norm.weight": compressor_norm_weight(),
            **{
                name: (w if bf16 else w.to(torch.float32)).contiguous()
                for name, w in weights.items()
            },
        }
    )

    with Graph(
        f"compressor_{ratio}_{int(bf16)}",
        input_types=[
            TensorType(DType.bfloat16, ["s", HIDDEN], DeviceRef.GPU())
        ],
    ) as graph:
        graph.output(module.compressor.project(graph.inputs[0].tensor))
    return session.load(graph, weights_registry=module.state_dict())


TESTDATA = Path(__file__).parent / "testdata"
PAGE = 128
TOP_K = 512
# The config fields the layer resolver reads.
_LAYOUT_FIELDS = (
    "num_hidden_layers",
    "compress_ratios",
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "candidate_source_layer_id",
    "engram_layer_ids",
    "index_topk",
    "rope_theta",
    "rope_scaling",
    "compress_rope_theta",
)


@dataclass(frozen=True)
class ListLayer:
    """What a trunk layer's index list reads."""

    layer_id: int
    ratio: int
    source: int | None
    """The index source whose selection it reads; ``None`` if window-only."""
    comp_index: int
    """Its KV source's index among the compressed leaf's layers."""


def trunk_list_layers() -> tuple[list[ListLayer], dict[str, tuple[int, int]]]:
    """Returns the checkpoint config's trunk layers and the leaves' geometry.

    The leaves are ``swa`` (every layer's window), ``c2a`` and ``c1a`` (the
    ratio-2 and ratio-1 KV sources' records), as ``(layers, page)``.
    """
    text = json.loads((TESTDATA / "config.json").read_text())["text_config"]
    rows = resolve(
        DeepseekV41Config(
            dtype=DType.bfloat16,
            kv_params=Mock(),
            devices=[],
            max_seq_len=0,
            **{name: text[name] for name in _LAYOUT_FIELDS},
        )
    )
    trunk = [row for row in rows if row.side != "drafter"]
    kv_sources = {
        ratio: sorted(
            {
                r.kv_source_layer_id
                for r in trunk
                if r.compress_ratio == ratio
                and r.kv_source_layer_id is not None
            }
        )
        for ratio in (1, 2)
    }
    layers = []
    for row in trunk:
        # Only compressed layers read a KV source.
        comp_index = 0
        if row.kv_source_layer_id is not None:
            comp_index = kv_sources[row.compress_ratio].index(
                row.kv_source_layer_id
            )
        layers.append(
            ListLayer(
                row.layer_id,
                row.compress_ratio,
                row.index_source_layer_id,
                comp_index,
            )
        )
    leaves = {
        "swa": (len(rows), PAGE),
        "c2a": (len(kv_sources[2]), PAGE // 2),
        "c1a": (len(kv_sources[1]), PAGE),
    }
    return layers, leaves


def index_lists_model(session: InferenceSession, fused: bool) -> Model:
    """A forward's trunk index lists, from :func:`index_list` or the ops.

    Every compressed layer reads its index source's selection of
    ``TOP_K`` key ids through ``key_positions``, as the model's ``end``
    does. Inputs: positions, request ids, one id selection per index
    source, then each leaf's lookup table and blocks.
    """
    layers, leaves = trunk_list_layers()
    sources = sorted(
        {layer.source for layer in layers if layer.source is not None}
    )
    ratio_of = {layer.layer_id: layer.ratio for layer in layers}
    gpu = DeviceRef.GPU()
    build = index_list if fused else index_list_ops
    with Graph(
        f"index_lists_{int(fused)}",
        input_types=[
            TensorType(DType.int32, ["T"], gpu),
            TensorType(DType.int32, ["T"], gpu),
            *(TensorType(DType.int32, ["T", TOP_K], gpu) for _ in sources),
            *(TensorType(DType.uint32, ["B", "P"], gpu) for _ in leaves),
            *(
                BufferType(DType.uint8, [2, 1, n, page, 1, 1], gpu)
                for n, page in leaves.values()
            ),
        ],
    ) as graph:
        positions, bid = graph.inputs[0].tensor, graph.inputs[1].tensor
        ids = [v.tensor for v in graph.inputs[2 : 2 + len(sources)]]
        tables = [v.tensor for v in graph.inputs[2 + len(sources) :][:3]]
        blocks = [v.buffer for v in graph.inputs[5 + len(sources) :]]
        zero = ops.constant(0, DType.uint32, gpu)
        leaf = {
            name: PagedCacheValues(
                kv_blocks=b,
                cache_lengths=ops.constant([0], DType.uint32, gpu),
                lookup_table=t,
                max_prompt_length=zero,
                max_cache_length=zero,
                page_stride=ops.constant([-1], DType.int64, DeviceRef.CPU()),
            )
            for name, b, t in zip(leaves, blocks, tables, strict=True)
        }
        selected = {
            s: key_positions(i, ratio_of[s])
            for s, i in zip(sources, ids, strict=True)
        }
        graph.output(
            *(
                build(
                    positions,
                    bid,
                    leaf["swa"],
                    layer.layer_id,
                    selected=(
                        None if layer.source is None else selected[layer.source]
                    ),
                    comp=(
                        leaf["c2a" if layer.ratio == 2 else "c1a"]
                        if layer.ratio
                        else None
                    ),
                    comp_layer=layer.comp_index,
                    ratio=layer.ratio,
                )
                for layer in layers
            )
        )
    return session.load(graph)


def index_list_inputs(
    requests: Sequence[tuple[int, int]], seed: int
) -> list[npt.NDArray[np.generic]]:
    """The inputs of :func:`index_lists_model` for ``(cached, new)`` requests.

    Tokens are request-major. Each token's selections spread ``TOP_K`` key
    ids over the groups closed at its position, as a long context's top-k
    does; the lookup tables name random distinct blocks.
    """
    layers, leaves = trunk_list_layers()
    sources = sorted(
        {layer.source for layer in layers if layer.source is not None}
    )
    ratio_of = {layer.layer_id: layer.ratio for layer in layers}
    rng = np.random.default_rng(seed)
    positions = np.concatenate([np.arange(c, c + n) for c, n in requests])
    bid = np.repeat(np.arange(len(requests)), [n for _, n in requests])
    arrays: list[npt.NDArray[np.generic]] = [
        positions.astype(np.int32),
        bid.astype(np.int32),
    ]
    for source in sources:
        visible = np.maximum((positions + 1) // ratio_of[source], 1)
        picks = rng.integers(0, 1 << 40, (len(positions), TOP_K))
        arrays.append((picks % visible[:, None]).astype(np.int32))
    pages = max(c + n for c, n in requests) // PAGE + 2
    for _ in leaves:
        arrays.append(
            np.stack(
                [rng.permutation(1 << 16)[:pages] for _ in requests]
            ).astype(np.uint32)
        )
    for n, page in leaves.values():
        arrays.append(np.zeros((2, 1, n, page, 1, 1), np.uint8))
    return arrays
