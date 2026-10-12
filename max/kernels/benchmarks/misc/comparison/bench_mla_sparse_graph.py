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
"""MAX's sparse MLA op over an FP8 latent cache under a CUDA graph.

Runs the op a DSA attention layer calls, `mla_prefill_decode_graph` with
sparse indices, on a paged cache set up by `PagedKVCacheManager`: `--batch`
requests with `--cache-len` tokens already cached and `--q-len` new tokens
each. A q length up to the decode limit takes the decode branch (speculative
verify); a longer one takes sparse prefill.

The graph holds `--copies` calls, one per cache layer, so one call cannot
reuse the previous call's keys from L2. The op is timed whole; its kernels
(RoPE, norm, cache store, Q absorb, attention, V up-projection) come apart
under `nsys --cuda-graph-trace=node`.

    br //max/kernels/benchmarks/misc/comparison:bench_mla_sparse_graph -- \
        --batch 8 --q-len 6 --cache-len 4096
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import time
from dataclasses import asdict, dataclass

import gpu_telemetry
import numpy as np
import sparse_mla_indices
import torch
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flare_mla_prefill_plan, mla_prefill_decode_graph
from max.nn.kv_cache import KVCacheInputsPerDevice, MLAKVCacheParams
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.modeling.types import RequestID

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 256
PAGE_SIZE = 128
BUFFER_TOK_SIZE = 16384
# The model passes no attention sink.
NO_SINK = -1.0e38


@dataclass
class Result:
    arm: str
    batch: int
    q_len: int
    cache_len: int
    rows: int
    copies: int
    replays: int
    us_per_op: float
    us_per_op_all: list[float]
    wall_start: float
    wall_end: float


def build_graph(
    kv_params: MLAKVCacheParams,
    num_heads: int,
    top_k: int,
    max_positions: int,
    copies: int,
) -> Graph:
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    depth = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    input_types = [
        TensorType(
            DType.bfloat16,
            ["total_tokens", num_heads, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM],
            gpu,
        ),
        TensorType(DType.bfloat16, ["total_tokens", depth], gpu),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        TensorType(DType.int32, ["total_tokens", top_k], gpu),
        TensorType(DType.bfloat16, [max_positions, QK_ROPE_HEAD_DIM], gpu),
        TensorType(DType.bfloat16, [KV_LORA_RANK], gpu),
        TensorType(
            DType.bfloat16, [num_heads * QK_NOPE_HEAD_DIM, KV_LORA_RANK], gpu
        ),
        TensorType(
            DType.bfloat16, [num_heads, KV_LORA_RANK, QK_NOPE_HEAD_DIM], gpu
        ),
        TensorType(DType.bfloat16, [num_heads, V_HEAD_DIM, KV_LORA_RANK], gpu),
        # The model binds the batch's page-aligned context length on the host.
        TensorType(DType.int32, [1], cpu),
        *tree.leaves(kv_params.get_symbolic_inputs()[0]),
    ]
    with Graph("mla_sparse_graph_bench", input_types=input_types) as g:
        (
            q,
            kv,
            row_offsets,
            indices,
            freqs,
            gamma,
            w_k,
            w_uk,
            w_uv,
            context_length,
        ) = (v.tensor for v in g.inputs[:10])
        kv_collection = kv_params.unflatten_kv_inputs(iter(g.inputs[10:]))[0]
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        buffer_row_offsets, cache_offsets, _ = flare_mla_prefill_plan(
            kv_params,
            row_offsets,
            kv_collection,
            ops.constant(0, DType.uint32, device=cpu),
            BUFFER_TOK_SIZE,
            max_chunks=1,
        )
        topk_lengths = ops.broadcast_to(
            ops.constant(top_k, DType.int32, device=gpu), (q.shape[0],)
        )
        sink = ops.broadcast_to(
            ops.constant(NO_SINK, DType.float32, device=gpu), (num_heads,)
        )
        outputs = [
            mla_prefill_decode_graph(
                q,
                kv,
                row_offsets,
                freqs,
                gamma,
                buffer_row_offsets,
                cache_offsets,
                context_length,
                w_k,
                w_uk,
                w_uv,
                kv_params,
                kv_collection,
                ops.constant(layer, DType.uint32, device=cpu),
                MHAMaskVariant.CAUSAL_MASK,
                (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5,
                1e-6,
                V_HEAD_DIM,
                kv_collection.attention_dispatch_metadata,
                kv_collection.mla_num_partitions,
                sparse_indices=indices,
                sparse_topk_lengths=topk_lengths,
                sparse_attn_sink=sink,
                sparse_indices_stride=top_k,
            )
            for layer in range(copies)
        ]
        g.output(*outputs)
    return g


def _bf16(shape: tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
    return torch.randn(shape, device="cuda").mul_(scale).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=6)
    parser.add_argument("--cache-len", default="1024,4096,75000")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument(
        "--copies",
        type=int,
        default=0,
        help="Op copies per graph; 0 sizes them to read 2x L2 per replay.",
    )
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--repeat-sleep-s",
        type=float,
        default=0.0,
        help=(
            "Idle the GPU this long before each timed repeat, so every repeat"
            " starts below the power limit."
        ),
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Label each case's timed replays with an NVTX range.",
    )
    parser.add_argument("--seed", type=int, default=0)
    # Tags kbench passes through from the shape YAML.
    parser.add_argument("--model", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()

    cache_lens = [int(c) for c in args.cache_len.split(",")]
    tokens_per_request = max(cache_lens) + args.q_len
    pages_per_request = -(-tokens_per_request // PAGE_SIZE)
    max_positions = pages_per_request * PAGE_SIZE

    positions_by_len = {
        cache_len: sparse_mla_indices.topk_positions(
            args.batch, args.q_len, cache_len, args.top_k, seed=args.seed
        )
        for cache_len in cache_lens
    }
    if args.copies == 0:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        args.copies = max(
            sparse_mla_indices.copies_to_exceed(
                positions,
                args.q_len,
                KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                2 * l2_bytes,
            )
            for positions in positions_by_len.values()
        )

    device = Accelerator()
    session = InferenceSession(devices=[device])
    kv_params = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        num_layers=args.copies,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=args.num_heads,
    )
    model = session.load(
        build_graph(
            kv_params, args.num_heads, args.top_k, max_positions, args.copies
        )
    )
    rows = args.batch * args.q_len
    weights = [
        _bf16((max_positions, QK_ROPE_HEAD_DIM), 1.0),
        _bf16((KV_LORA_RANK,), 1.0),
        _bf16((args.num_heads * QK_NOPE_HEAD_DIM, KV_LORA_RANK)),
        _bf16((args.num_heads, KV_LORA_RANK, QK_NOPE_HEAD_DIM)),
        _bf16((args.num_heads, V_HEAD_DIM, KV_LORA_RANK)),
    ]
    results = []
    for key, cache_len in enumerate(cache_lens, start=1):
        kv_manager = PagedKVCacheManager(
            params=kv_params,
            session=session,
            total_num_pages=args.batch * pages_per_request,
            max_batch_size=args.batch,
        )
        batch = []
        for _ in range(args.batch):
            ctx = TextContext(
                request_id=RequestID(),
                max_length=tokens_per_request + 1,
                tokens=TokenBuffer(
                    np.zeros(cache_len + args.q_len, dtype=np.int64)
                ),
            )
            kv_manager.claim(ctx)
            # Reach `cache_len` cached tokens and `q_len` new ones the way
            # chunked prefill does: allocate and commit a first chunk.
            if cache_len:
                ctx.tokens.chunk(cache_len)
                kv_manager.alloc(ctx)
                ctx.tokens.advance_chunk()
            kv_manager.alloc(ctx)
            batch.append(ctx)
        kv_inputs = kv_manager.runtime_inputs([batch])
        assert isinstance(kv_inputs, tuple)
        per_device = kv_inputs[0]
        assert isinstance(per_device, KVCacheInputsPerDevice)
        cache_lengths = (
            torch.from_dlpack(per_device.cache_lengths).cpu().numpy()
        )
        if not (cache_lengths == cache_len).all():
            raise RuntimeError(f"cache lengths {cache_lengths} != {cache_len}")
        blocks = torch.from_dlpack(per_device.kv_blocks)
        blocks.copy_(
            (torch.randn(blocks.shape, device="cuda") * 0.5).to(blocks.dtype)
        )

        positions = positions_by_len[cache_len]
        step = [
            _bf16(
                (rows, args.num_heads, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), 1.0
            ),
            _bf16((rows, KV_LORA_RANK + QK_ROPE_HEAD_DIM), 1.0),
            torch.from_numpy(
                np.arange(0, rows + 1, args.q_len, dtype=np.uint32)
            ).cuda(),
            torch.from_numpy(positions).cuda(),
        ]
        aligned = -(-(cache_len + args.q_len) // PAGE_SIZE) * PAGE_SIZE
        context_length = np.array([args.batch * aligned], dtype=np.int32)
        inputs = [
            *(Buffer.from_dlpack(t) for t in step + weights),
            Buffer.from_dlpack(torch.from_numpy(context_length)),
            *tree.leaves(kv_inputs),
        ]
        outputs = model.capture(key, *inputs)
        for _ in range(3):
            model.replay(key, *inputs)
        torch.cuda.synchronize()
        label = f"b{args.batch}_q{args.q_len}_l{cache_len}"
        wall_start = time.time()
        per_op = gpu_telemetry.time_bursts(
            functools.partial(model.replay, key, *inputs),
            args.copies,
            args.replays,
            args.repeats,
            args.repeat_sleep_s,
            label if args.nvtx else "",
        )
        wall_end = time.time()
        for out in outputs:
            if not torch.isfinite(torch.from_dlpack(out).float()).all():
                raise RuntimeError("non-finite output")
        result = Result(
            arm="max_mla_sparse",
            batch=args.batch,
            q_len=args.q_len,
            cache_len=cache_len,
            rows=rows,
            copies=args.copies,
            replays=args.replays,
            us_per_op=statistics.median(per_op),
            us_per_op_all=per_op,
            wall_start=wall_start,
            wall_end=wall_end,
        )
        print(json.dumps(asdict(result)), flush=True)
        results.append(asdict(result))
        for ctx in batch:
            kv_manager.release(ctx)
        del kv_manager, outputs, inputs
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
