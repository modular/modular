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
"""Checks paged attention under the relative-logits mask against PyTorch.

The bias is gathered by `d = q_pos - k_pos` from a `(tokens, heads, extent)`
table and added on every visible position with `0 <= d < extent`.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import flash_attention_ragged
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kv_cache import MHAKVCacheParams
from test_common.simple_kv_cache import paged_kv_cache_inputs

HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE = 8, 2, 64, 128
SCALE = 1.0 / math.sqrt(HEAD_DIM)


@dataclass(frozen=True)
class Case:
    prompt_lens: list[int]
    cache_lengths: list[int]
    extent: int
    window: int = -1


CASES = {
    "causal": Case([17, 40], [0, 0], extent=64),
    # Distances past the extent carry no bias.
    "short_extent": Case([33, 9], [0, 0], extent=8),
    # Cached tokens shift each query's position past its bias row.
    "cached": Case([5, 12], [20, 3], extent=32),
    "window": Case([48], [0], extent=16, window=8),
    # One token per sequence runs the decode kernels; the long prefix spans
    # several pages.
    "decode": Case([1, 1], [300, 40], extent=16),
    "decode_window": Case([1, 1], [258, 7], extent=8, window=6),
}


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _build(case: Case) -> Graph:
    kv_params = _kv_params()
    with Graph(
        "relative_logits",
        input_types=[
            TensorType(
                DType.bfloat16, ["tokens", HEADS, HEAD_DIM], DeviceRef.GPU()
            ),
            TensorType(DType.uint32, ["batch_plus_1"], DeviceRef.GPU()),
            TensorType(
                DType.bfloat16, ["tokens", HEADS, case.extent], DeviceRef.GPU()
            ),
            *kv_params.flattened_kv_inputs(),
        ],
    ) as g:
        q, offsets, bias = (x.tensor for x in g.inputs[:3])
        kv_collection = kv_params.unflatten_kv_inputs(iter(g.inputs[3:]))[0]
        out = flash_attention_ragged(
            kv_params,
            input=q,
            input_row_offsets=offsets,
            kv_collection=kv_collection,
            layer_idx=ops.constant(0, DType.uint32, DeviceRef.CPU()),
            mask_variant=(
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if case.window > 0
                else MHAMaskVariant.CAUSAL_MASK
            ),
            scale=SCALE,
            local_window_size=case.window,
            rel_logits=bias,
        )
        g.output(out.cast(DType.float32))
    return g


@pytest.fixture(scope="module")
def models() -> dict[str, Model]:
    # Compile every graph before touching the device, so the compiles can be
    # recorded on CPU.
    session = InferenceSession(devices=[Accelerator()])
    return {name: session.load(_build(case)) for name, case in CASES.items()}


def _reference(
    q: torch.Tensor,
    k: list[torch.Tensor],
    v: list[torch.Tensor],
    bias: torch.Tensor,
    offsets: list[int],
    case: Case,
) -> torch.Tensor:
    group = HEADS // KV_HEADS
    out = torch.empty_like(q)
    for r, cache_len in enumerate(case.cache_lengths):
        rows = slice(offsets[r], offsets[r + 1])
        qr = q[rows]  # [P, H, D]
        kr = k[r].repeat_interleave(group, dim=1)  # [L, H, D]
        vr = v[r].repeat_interleave(group, dim=1)
        scores = torch.einsum("phd,lhd->hpl", qr, kr) * SCALE
        q_pos = cache_len + torch.arange(qr.shape[0])[:, None]
        k_pos = torch.arange(kr.shape[0])[None, :]
        dist = q_pos - k_pos
        visible = dist >= 0
        if case.window > 0:
            visible &= dist < case.window
        in_extent = visible & (dist < case.extent)
        gathered = torch.gather(
            bias[rows].permute(1, 0, 2),  # [H, P, extent]
            2,
            dist.clamp(0, case.extent - 1)[None].expand(HEADS, -1, -1),
        )
        scores = scores + torch.where(in_extent, gathered, 0.0)
        scores = scores.masked_fill(~visible, -math.inf)
        out[rows] = torch.einsum("hpl,lhd->phd", scores.softmax(-1), vr)
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_relative_logits(models: dict[str, Model], name: str) -> None:
    case = CASES[name]
    torch.manual_seed(0)
    kv_params = _kv_params()
    inputs = paged_kv_cache_inputs(
        kv_params, case.prompt_lens, cache_lengths=case.cache_lengths
    )
    # The helper zero-fills the pages; give them values to attend over.
    blocks = torch.randn(
        *inputs.kv_blocks.shape, dtype=torch.bfloat16, device="cuda"
    )
    inputs = dataclasses.replace(inputs, kv_blocks=Buffer.from_dlpack(blocks))

    offsets = np.cumsum([0, *case.prompt_lens]).tolist()
    tokens = offsets[-1]
    q = torch.randn(tokens, HEADS, HEAD_DIM, dtype=torch.bfloat16)
    bias = torch.randn(tokens, HEADS, case.extent, dtype=torch.bfloat16)

    out = models[name].execute(
        Buffer.from_dlpack(q).to(Accelerator()),
        Buffer.from_numpy(np.array(offsets, np.uint32)).to(Accelerator()),
        Buffer.from_dlpack(bias).to(Accelerator()),
        *tree.leaves(inputs),
    )[0]
    assert isinstance(out, Buffer)

    # Pages go to requests in order; read each request's keys and values.
    cpu_blocks = blocks.cpu().float()
    k, v, page = [], [], 0
    for cache_len, prompt_len in zip(
        case.cache_lengths, case.prompt_lens, strict=True
    ):
        length = cache_len + prompt_len
        pages = -(-length // PAGE_SIZE)
        flat = cpu_blocks[page : page + pages, :, 0].transpose(0, 1)
        flat = flat.reshape(2, pages * PAGE_SIZE, KV_HEADS, HEAD_DIM)
        k.append(flat[0, :length])
        v.append(flat[1, :length])
        page += pages

    expected = _reference(q.float(), k, v, bias.float(), offsets, case)
    np.testing.assert_allclose(
        out.to(CPU()).to_numpy(), expected.numpy(), rtol=2e-2, atol=2e-2
    )
