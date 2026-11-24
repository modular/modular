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
"""FlashInfer's TRT-LLM-gen FP8 sparse MLA kernel under a CUDA graph.

The competitor arm for `bench_mla_sparse_graph.py`. vLLM's default backend
for DSA sparse attention over an FP8 latent cache on SM100
(`FLASHINFER_MLA_SPARSE`) runs this kernel with one query row per token, a
64-token page and physical row indices. Run it with the Python of a venv that
has the `vllm` wheel installed, so FlashInfer is the version vLLM pins.

The graph holds `--copies` calls, each over its own cache, so one call cannot
reuse the previous call's keys from L2. PDL is FlashInfer's default on SM100.

    <venv>/bin/python bench_flashinfer_trtllm_mla_sparse.py --batch 8 --q-len 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import gpu_telemetry
import numpy as np
import sparse_mla_indices
import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
PAGE_SIZE = 64
WORKSPACE_BYTES = 394 << 20


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
    kernels: list[str]
    wall_start: float
    wall_end: float


def _rand_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape, device="cuda") * 0.5).to(torch.float8_e4m3fn)


def make_page_table(
    batch: int, tokens_per_request: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Scatters each request's pages over the cache, as a paged allocator does."""
    pages_per_request = -(-tokens_per_request // PAGE_SIZE)
    total = batch * pages_per_request
    return rng.permutation(total).reshape(batch, pages_per_request), total


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
        help="Calls per graph; 0 sizes them to read 2x L2 per replay.",
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
        "--log-kernels",
        action="store_true",
        help="Record the kernels one replay runs.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Label each case's timed replays with an NVTX range.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()

    depth = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    sm_scale = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5
    cache_lens = [int(c) for c in args.cache_len.split(",")]
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
                positions, args.q_len, depth, 2 * l2_bytes
            )
            for positions in positions_by_len.values()
        )
    rows = args.batch * args.q_len
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.int8, device="cuda")
    results = []
    for cache_len in cache_lens:
        rng = np.random.default_rng(args.seed)
        page_table, num_pages = make_page_table(
            args.batch, cache_len + args.q_len, rng
        )
        positions = positions_by_len[cache_len]
        block_tables = torch.from_numpy(
            sparse_mla_indices.physical_rows(
                positions, page_table, args.q_len, PAGE_SIZE
            )
        ).cuda()[:, None, :]
        seq_lens = torch.from_numpy(
            sparse_mla_indices.valid_counts(positions)
        ).cuda()
        q = _rand_fp8((rows, 1, args.num_heads, depth))
        caches = [
            _rand_fp8((num_pages, 1, PAGE_SIZE, depth))
            for _ in range(args.copies)
        ]

        def call(
            kv: torch.Tensor,
            q: torch.Tensor = q,
            block_tables: torch.Tensor = block_tables,
            seq_lens: torch.Tensor = seq_lens,
        ) -> torch.Tensor:
            return trtllm_batch_decode_with_kv_cache_mla(
                query=q,
                kv_cache=kv,
                workspace_buffer=workspace,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=args.top_k,
                sparse_mla_top_k=args.top_k,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
            )

        for kv in caches:
            call(kv)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [call(kv) for kv in caches]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        label = f"b{args.batch}_q{args.q_len}_l{cache_len}"
        wall_start = time.time()
        per_op = gpu_telemetry.time_bursts(
            graph.replay,
            args.copies,
            args.replays,
            args.repeats,
            args.repeat_sleep_s,
            label if args.nvtx else "",
        )
        wall_end = time.time()
        kernels = (
            gpu_telemetry.replay_kernels(graph.replay)
            if args.log_kernels
            else []
        )
        for out in outputs:
            if not torch.isfinite(out.float()).all():
                raise RuntimeError("non-finite output")
        result = Result(
            arm="flashinfer_trtllm_mla_sparse",
            batch=args.batch,
            q_len=args.q_len,
            cache_len=cache_len,
            rows=rows,
            copies=args.copies,
            replays=args.replays,
            us_per_op=statistics.median(per_op),
            us_per_op_all=per_op,
            kernels=kernels,
            wall_start=wall_start,
            wall_end=wall_end,
        )
        print(json.dumps(asdict(result)), flush=True)
        results.append(asdict(result))
        del graph, outputs, caches
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
