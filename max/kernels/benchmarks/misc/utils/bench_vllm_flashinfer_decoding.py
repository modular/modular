# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

# This python script benchmarks vll-flash-attention and flashinfer's decoding
# kernel in token generation.
#
# Usage:
#     python3 bench_vllm_flashinfer_decoding.py --backend all \
#             --cache-lens 512 1024 2048 --batch-size 32 64 128 256 \
#             --num-query-heads 32 --num-kv-heads 8 --dump-csv
#
# Note that one can set --flashinfer-tensorcore to specify using tensor core
# with flashinfer, which gives better perf for large BS and cache_len.
#
# Prerequisite:
# Install vllm-flash-attn and flashinfer via
#     pip install vllm_flash_attn
#     pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
#
# Ported from https://github.com/cli99/llm-scripts

import argparse
import csv
import gc
import random
import time

import flashinfer
import numpy as np
import torch
from flash_attn_interface import flash_attn_with_kvcache

NUM_BLOCKS = 32768
CSV_FILE = "flashattn_vs_flash_infer_decoding.csv"

STR_DTYPE_TO_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def seed_everything(seed) -> None:  # noqa: ANN001
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def bench(
    backend: str,
    num_seqs: int,
    cache_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
    kv_cache_dtype: str | None = None,
    flashinfer_tensorcore: bool = False,
) -> float:
    seed_everything(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(
        num_seqs, num_query_heads, head_size, dtype=dtype, device=device
    )
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0

    # Create the block tables.
    max_num_blocks_per_seq = (cache_len + block_size - 1) // block_size
    block_tables_lst: list[list[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(
        block_tables_lst, dtype=torch.int, device=device
    )

    # kv cache
    key_value_cache = torch.randn(
        NUM_BLOCKS,
        2,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # kv cachen length
    kv_lens = [cache_len for _ in range(num_seqs)]
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int, device=device)

    if backend == "flashinfer":
        qo_indptr = [0]
        kv_indptr = [0]
        kv_indices = []
        kv_last_page_lens = []
        for i in range(num_seqs):
            assert cache_len > 0

            num_blocks = (cache_len + block_size - 1) // block_size
            kv_indices.extend(block_tables[i, :num_blocks])
            kv_indptr.append(kv_indptr[-1] + num_blocks)
            kv_last_page_len = cache_len % block_size
            if kv_last_page_len == 0:
                kv_last_page_len = block_size
            kv_last_page_lens.append(kv_last_page_len)

        qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=device)
        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
        kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device=device)
        kv_last_page_lens = torch.tensor(
            kv_last_page_lens, dtype=torch.int32, device=device
        )
        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )

        # Context encoding, or prefill in flashinfer terms
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            "NHD",
            use_tensor_cores=flashinfer_tensorcore,
        )
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            "NONE",
            data_type=dtype,
        )
    else:
        wrapper = None

    def run_cuda_benchmark(
        num_iters: int,
        profile: bool = False,
    ) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if backend == "flashattn":
                if torch.cuda.get_device_capability()[0] == 9:
                    _ = flash_attn_with_kvcache(
                        q=query.unsqueeze(1),  # [BS, 1, num_heads, depth]
                        k_cache=key_cache,
                        v_cache=value_cache,
                        softmax_scale=scale,
                        causal=True,
                        page_table=block_tables,
                        cache_seqlens=kv_lens_tensor,
                        softcap=0,
                        window_size=(-1, -1),
                    ).squeeze(1)
                else:
                    _ = flash_attn_with_kvcache(
                        q=query.unsqueeze(1),  # [BS, 1, num_heads, depth]
                        k_cache=key_cache,
                        v_cache=value_cache,
                        softmax_scale=scale,
                        causal=True,
                        block_table=block_tables,
                        cache_seqlens=kv_lens_tensor,
                        softcap=0,
                        window_size=(-1, -1),
                    ).squeeze(1)
            elif backend == "flashinfer":
                _ = wrapper.forward(query, key_value_cache)

            else:
                raise ValueError(f"Unsupported backend: {backend}")
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return (end_time - start_time) / num_iters

    # Warmup.
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    return latency


@torch.inference_mode()
def main(
    backend: str,
    num_seqs_list: list[int],
    cache_len_list: list[int],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
    kv_cache_dtype: str | None = None,
    dump_csv: bool = False,
    flashinfer_tensorcore: bool = False,
) -> float:
    results = []
    for num_seqs in num_seqs_list:
        for cache_len in cache_len_list:
            t1, t2 = None, None
            if backend in ["flashattn", "all"]:
                t1 = bench(
                    backend="flashattn",
                    num_seqs=num_seqs,
                    cache_len=cache_len,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    block_size=block_size,
                    dtype=dtype,
                    seed=seed,
                    do_profile=do_profile,
                    device=device,
                    kv_cache_dtype=kv_cache_dtype,
                    flashinfer_tensorcore=flashinfer_tensorcore,
                )
                print(f"Flashattn kernel running time: {t1 * 1000000:.3f} us")

            gc.collect()
            torch.cuda.empty_cache()

            if backend in ["flashinfer", "all"]:
                t2 = bench(
                    backend="flashinfer",
                    num_seqs=num_seqs,
                    cache_len=cache_len,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    block_size=block_size,
                    dtype=dtype,
                    seed=seed,
                    do_profile=do_profile,
                    device=device,
                    kv_cache_dtype=kv_cache_dtype,
                    flashinfer_tensorcore=flashinfer_tensorcore,
                )
                print(f"Flashinfer kernel running time: {t2 * 1000000:.3f} us")

                line = [
                    num_seqs,
                    cache_len,
                    t1 * 1000000 if t1 else -1,
                    t2 * 1000000 if t2 else -1,
                    t1 / t2 if t1 and t2 else -1,
                ]
                results.append(line)
    if dump_csv:
        # Write the dictionary to a CSV file
        with open(CSV_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Batch Size",
                    "Seq Length",
                    "flashattn",
                    "flashinfer",
                    "Speedup",
                ]
            )
            for row in results:
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel."
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--cache-lens", nargs="+", type=int, default=[4096])
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[64, 80, 96, 112, 120, 128, 192, 256],
        default=128,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["flashattn", "flashinfer", "all"],
        default="flashattn",
    )
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float"],
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dump-csv", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help=(
            "Data type for kv cache storage. If 'auto', will use model "
            "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
            "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)"
        ),
    )
    parser.add_argument("--flashinfer-tensorcore", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    main(
        backend=args.backend,
        num_seqs_list=args.batch_sizes,
        cache_len_list=args.cache_lens,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
        kv_cache_dtype=args.kv_cache_dtype,
        dump_csv=args.dump_csv,
        flashinfer_tensorcore=args.flashinfer_tensorcore,
    )

# python bench_attn.py --backend all --cache-lens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 --batch-size 1 2 4 8 16 32 64 128 256 --dump-csv
