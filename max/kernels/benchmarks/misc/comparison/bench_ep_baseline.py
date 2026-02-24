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

# Expert Parallelism (EP) baseline benchmark for MAX.
# Run via Bazel: br //Kernels/benchmarks/comparison:bench_ep_baseline
#
# This script establishes baseline performance metrics for MAX EP dispatch/combine operations.
# Supports two modes:
#   1. Separate kernels (default): Times dispatch_async, dispatch_wait, combine_async, combine_wait
#   2. Fused kernels (--oneshot-ep): Times fused dispatch and combine kernels
# Reports effective GB/s for each phase.

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

# Import bench utilities from Bazel dependency (bench_utils target)
from bench import bench_kineto_with_cupti_warmup

# MAX imports
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

# MAX EP APIs
from max.nn.comm.ep.ep_config import EPConfig
from max.nn.comm.ep.ep_manager import EPBatchManager, EPCommInitializer


def _torch_dtype_from_max(dtype: DType) -> torch.dtype:
    if dtype == DType.bfloat16:
        return torch.bfloat16
    if dtype == DType.float16:
        return torch.float16
    if dtype == DType.float32:
        return torch.float32
    # Default to bf16 for unsupported combos here (FP8 inputs are quantized in kernels)
    return torch.bfloat16


@dataclass
class EPBenchmarkArgs:
    num_tokens: int
    hidden: int
    num_topk: int
    num_experts: int
    dispatch_dtype: DType
    combine_dtype: DType
    iters: int
    warmup: int
    gpus_per_node: int
    nodes: int
    max_tokens_per_rank: int | None
    profile: bool
    oneshot_ep: bool = False


def build_ep_graph(config: EPConfig, oneshot_ep: bool = False) -> Graph:
    """
    Build a MAX Graph that performs EP dispatch and combine operations.

    When oneshot_ep=False (default):
      - ep_dispatch_async on all local GPUs
      - ep_dispatch_wait to gather per-GPU received tokens
      - (no-op expert compute placeholder)
      - ep_combine_async to return tokens
      - ep_combine_wait to reconstruct per-device outputs

    When oneshot_ep=True:
      - ep_dispatch (fused) on all local GPUs
      - ep_combine (fused) to return and reconstruct tokens

    The graph returns one small reduction per device to avoid large host copies.
    """
    manager = EPBatchManager(config)

    # Build input types for the static EP comm buffers/pointers
    ep_static_input_types = []
    for t in manager.input_types():
        # Graph accepts both BufferType and TensorType in input_types
        ep_static_input_types.append(t)

    # Per-GPU dynamic inputs: tokens, topk, and router weights
    token_types: list[TensorType] = []
    topk_types: list[TensorType] = []
    router_weight_types: list[TensorType] = []
    for dev_id in range(config.n_gpus_per_node):
        token_types.append(
            TensorType(
                dtype=config.dispatch_dtype
                if not config.dispatch_dtype.is_float8()
                else DType.bfloat16,
                shape=[config.max_tokens_per_rank, config.hidden_size],
                device=DeviceRef.GPU(dev_id),
            )
        )
        topk_types.append(
            TensorType(
                dtype=DType.int32,
                shape=[config.max_tokens_per_rank, config.top_k],
                device=DeviceRef.GPU(dev_id),
            )
        )
        router_weight_types.append(
            TensorType(
                dtype=DType.float32,
                shape=[config.max_tokens_per_rank, config.top_k],
                device=DeviceRef.GPU(dev_id),
            )
        )

    with Graph(
        "ep_bench",
        input_types=[
            *ep_static_input_types,
            *token_types,
            *topk_types,
            *router_weight_types,
        ],
    ) as g:
        # Slice inputs
        total_static = len(ep_static_input_types)
        static_vals = g.inputs[:total_static]
        dyn_vals = g.inputs[total_static:]
        num_dyn = len(dyn_vals)
        assert (
            num_dyn == 3 * config.n_gpus_per_node
        )  # tokens, topk, router_weights

        # Let manager parse the EP buffer pointers and counters
        manager.fetch_buffers(static_vals)

        tokens_vals: list[TensorValue] = []
        topk_vals: list[TensorValue] = []
        router_weights_vals: list[TensorValue] = []
        for i in range(config.n_gpus_per_node):
            tokens_vals.append(
                dyn_vals[i].tensor
            )  # (max_tokens_per_rank, hidden)
            topk_vals.append(
                dyn_vals[config.n_gpus_per_node + i].tensor
            )  # (max_tokens_per_rank, top_k)
            router_weights_vals.append(
                dyn_vals[2 * config.n_gpus_per_node + i].tensor
            )  # (max_tokens_per_rank, top_k)

        # Dispatch on each device
        dispatched: list[tuple[TensorValue, ...]] = []
        if oneshot_ep:
            for dev_id in range(config.n_gpus_per_node):
                dispatched.append(
                    manager.ep_dispatch(
                        tokens_vals[dev_id], topk_vals[dev_id], device_id=dev_id
                    )
                )
        else:
            for dev_id in range(config.n_gpus_per_node):
                manager.ep_dispatch_async(
                    tokens_vals[dev_id], topk_vals[dev_id], device_id=dev_id
                )

            # Gather results
            for dev_id in range(config.n_gpus_per_node):
                dispatched.append(manager.ep_dispatch_wait(device_id=dev_id))
                # dispatched entries (non-FP8): (recv_tokens, row_offsets, expert_ids, stats)
                # we ignore stats/ids here; feed recv_tokens back to combine below

        # Combine on each device; use the first tensor returned by dispatch_wait as combine input
        outputs = []
        if oneshot_ep:
            for dev_id in range(config.n_gpus_per_node):
                recv_tokens = dispatched[dev_id][0]  # tokens to send back
                out = manager.ep_combine(
                    recv_tokens, router_weights_vals[dev_id], device_id=dev_id
                )  # (num_tokens, hidden)
                # Reduce to a small scalar per device (sum) to minimize host transfer
                outputs.append(ops.sum(out))
        else:
            for dev_id in range(config.n_gpus_per_node):
                recv_tokens = dispatched[dev_id][0]  # tokens to send back
                manager.ep_combine_async(recv_tokens, device_id=dev_id)

            # Complete combine
            for dev_id in range(config.n_gpus_per_node):
                out = manager.ep_combine_wait(
                    router_weights_vals[dev_id], device_id=dev_id
                )  # (num_tokens, hidden)
                # Reduce to a small scalar per device (sum) to minimize host transfer
                # Keep reduction on-device for profiling clarity
                outputs.append(ops.sum(out))

        g.output(*outputs)
        return g


def make_inputs_for_execute(
    config: EPConfig,
    initializer: EPCommInitializer,
    dispatch_torch_dtype: torch.dtype,
) -> list[Buffer]:
    """
    Prepare input tensors for model.execute():
      - EP static tensors (atomic counters, send/recv pointers) from initializer
      - Per-device tokens (max_tokens_per_rank, hidden) on each GPU
      - Per-device topk (max_tokens_per_rank, top_k) on each GPU
      - Per-device router weights (max_tokens_per_rank, top_k) on each GPU

    Note: We convert torch tensors to MAX Tensors with the correct CUDA device
    context active to avoid DLPack export issues.
    """
    inputs: list[Buffer] = []
    # Static EP inputs
    inputs.extend(initializer.model_inputs())

    # First append all token tensors (grouped, not interleaved)
    for dev_id in range(config.n_gpus_per_node):
        with torch.cuda.device(dev_id):
            x = torch.randn(
                (config.max_tokens_per_rank, config.hidden_size),
                dtype=dispatch_torch_dtype,
                device=f"cuda:{dev_id}",
            )
            inputs.append(Buffer.from_dlpack(x))

    # Then append all topk tensors
    for dev_id in range(config.n_gpus_per_node):
        with torch.cuda.device(dev_id):
            topk = torch.randint(
                low=0,
                high=config.n_experts,
                size=(config.max_tokens_per_rank, config.top_k),
                dtype=torch.int32,
                device=f"cuda:{dev_id}",
            )
            inputs.append(Buffer.from_dlpack(topk))

    # Finally append router weights (uniform weights for benchmark)
    for dev_id in range(config.n_gpus_per_node):
        with torch.cuda.device(dev_id):
            router_weights = (
                torch.ones(
                    (config.max_tokens_per_rank, config.top_k),
                    dtype=torch.float32,
                    device=f"cuda:{dev_id}",
                )
                / config.top_k
            )
            inputs.append(Buffer.from_dlpack(router_weights))

    return inputs


def compute_bytes_per_token(
    hidden: int,
    dtype: DType,
    fp8_scale_bytes: int = 4,
) -> float:
    """
    Approximate bytes per token transferred.
    - BF16/FP16: hidden * 2 or hidden * 2
    - FP8 blockwise (1x128): hidden * 1 + (hidden / 128) * fp8_scale_bytes
    """
    if dtype.is_float8():
        return float(hidden) + float(hidden // 128) * fp8_scale_bytes
    if dtype == DType.bfloat16 or dtype == DType.float16:
        return float(hidden * 2)
    if dtype == DType.float32:
        return float(hidden * 4)
    # Fallback
    return float(hidden * 2)


def run_bench_max_ep(args: EPBenchmarkArgs) -> None:
    # Determine GPU topology
    visible_gpus = torch.cuda.device_count()
    assert visible_gpus >= 1, "CUDA device not found"
    n_gpus = args.gpus_per_node or visible_gpus
    assert n_gpus <= visible_gpus, (
        f"Requested {n_gpus} GPUs, but only {visible_gpus} visible"
    )

    # Prepare EP config
    max_tokens_per_rank = args.max_tokens_per_rank or args.num_tokens
    config = EPConfig(
        dispatch_dtype=args.dispatch_dtype,
        combine_dtype=args.combine_dtype,
        hidden_size=args.hidden,
        top_k=args.num_topk,
        n_experts=args.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_gpus,
        n_nodes=args.nodes,
    )

    # Session with all local GPUs
    session = InferenceSession(devices=[Accelerator(i) for i in range(n_gpus)])

    # Initialize NVSHMEM buffers
    initializer = EPCommInitializer(config)
    initializer.ep_init(session)

    # Build and compile EP bench graph
    graph = build_ep_graph(config, oneshot_ep=args.oneshot_ep)
    model = session.load(graph)

    # Prepare runtime inputs
    dispatch_torch_dtype = _torch_dtype_from_max(
        args.dispatch_dtype
        if not args.dispatch_dtype.is_float8()
        else DType.bfloat16
    )
    execute_inputs = make_inputs_for_execute(
        config, initializer, dispatch_torch_dtype
    )

    # Run with Kineto timing for EP kernel names
    def run_once() -> list[Buffer]:
        return model.execute(*execute_inputs)

    # Compute effective bandwidth per device (bytes / avg_time)
    bytes_per_token_dispatch = compute_bytes_per_token(
        args.hidden, args.dispatch_dtype
    )
    bytes_per_token_combine = compute_bytes_per_token(
        args.hidden, args.combine_dtype
    )
    # Total selections per token = top_k
    dispatch_bytes = args.num_tokens * args.num_topk * bytes_per_token_dispatch
    combine_bytes = args.num_tokens * args.num_topk * bytes_per_token_combine

    print("=" * 80)
    print(
        f"MAX EP Benchmark (tokens={args.num_tokens}, hidden={args.hidden}, top_k={args.num_topk}, experts={args.num_experts}, gpus={n_gpus})"
    )
    print(
        f"Mode: {'fused (oneshot)' if args.oneshot_ep else 'separate (async/wait)'}"
    )
    print("=" * 80)
    print(f"{'Phase':<20} {'Avg time (ms)':<15} {'GB/s (per device)':<20}")

    if args.oneshot_ep:
        # Fused kernels: single dispatch and combine kernel each
        time_dispatch = bench_kineto_with_cupti_warmup(
            run_once,
            kernel_names="shmem_ep_comm_dispatch_kernel",
            num_tests=args.iters,
            suppress_kineto_output=not args.profile,
            with_multiple_kernels=True,
        )
        assert isinstance(time_dispatch, float)

        time_combine = bench_kineto_with_cupti_warmup(
            run_once,
            kernel_names="shmem_ep_comm_combine_kernel",
            num_tests=args.iters,
            suppress_kineto_output=not args.profile,
            with_multiple_kernels=True,
        )
        assert isinstance(time_combine, float)

        dispatch_gbps = (dispatch_bytes / 1e9) / time_dispatch
        combine_gbps = (combine_bytes / 1e9) / time_combine

        total_bytes = dispatch_bytes + combine_bytes
        total_time_s = time_dispatch + time_combine
        total_gbps = (total_bytes / 1e9) / total_time_s

        print(
            f"{'dispatch (fused)':<20} {time_dispatch * 1e3:<15.3f} {dispatch_gbps:<20.2f}"
        )
        print(
            f"{'combine (fused)':<20} {time_combine * 1e3:<15.3f} {combine_gbps:<20.2f}"
        )
        print("-" * 80)
        print(
            f"{'dispatch+combine':<20} {total_time_s * 1e3:<15.3f} {total_gbps:<20.2f}"
        )
    else:
        # Separate kernels: dispatch_async, dispatch_wait, combine_async, combine_wait
        times_dispatch = bench_kineto_with_cupti_warmup(
            run_once,
            kernel_names=(
                "shmem_ep_comm_dispatch_async_k",
                "shmem_ep_comm_dispatch_wait_ke",
            ),
            num_tests=args.iters,
            suppress_kineto_output=not args.profile,
            with_multiple_kernels=True,
        )
        assert isinstance(times_dispatch, tuple)

        times_combine = bench_kineto_with_cupti_warmup(
            run_once,
            kernel_names=(
                "shmem_ep_comm_combine_async_ke",
                "shmem_ep_comm_combine_wait_ker",
            ),
            num_tests=args.iters,
            suppress_kineto_output=not args.profile,
            with_multiple_kernels=True,
        )
        assert isinstance(times_combine, tuple)

        dispatch_async_gbps = (dispatch_bytes / 1e9) / times_dispatch[0]
        dispatch_wait_gbps = (dispatch_bytes / 1e9) / times_dispatch[1]
        combine_async_gbps = (combine_bytes / 1e9) / times_combine[0]
        combine_wait_gbps = (combine_bytes / 1e9) / times_combine[1]

        total_bytes = dispatch_bytes + combine_bytes
        total_time_s = sum(times_dispatch) + sum(times_combine)
        total_gbps = (total_bytes / 1e9) / (
            total_time_s / 2.0
        )  # divide by 2 to average send/wait overlap conservatively

        print(
            f"{'dispatch_async':<20} {times_dispatch[0] * 1e3:<15.3f} {dispatch_async_gbps:<20.2f}"
        )
        print(
            f"{'dispatch_wait':<20} {times_dispatch[1] * 1e3:<15.3f} {dispatch_wait_gbps:<20.2f}"
        )
        print(
            f"{'combine_async':<20} {times_combine[0] * 1e3:<15.3f} {combine_async_gbps:<20.2f}"
        )
        print(
            f"{'combine_wait':<20} {times_combine[1] * 1e3:<15.3f} {combine_wait_gbps:<20.2f}"
        )
        print("-" * 80)
        print(f"{'dispatch+combine':<20} {'~':<15} {total_gbps:<20.2f}")

    print("=" * 80)


def parse_args() -> EPBenchmarkArgs:
    parser = argparse.ArgumentParser(
        description="MAX Expert Parallelism benchmark"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=128,
        help="Number of tokens per device",
    )
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden size")
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of experts per token"
    )
    parser.add_argument(
        "--num-experts", type=int, default=64, help="Total number of experts"
    )
    parser.add_argument(
        "--dispatch-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Dispatch activation dtype (FP8 path can be added later)",
    )
    parser.add_argument(
        "--combine-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Combine activation dtype",
    )
    parser.add_argument(
        "--iters", type=int, default=30, help="Number of test iterations"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (unused; kineto handles warmup)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=0,
        help="GPUs to use on this node (0 = all visible)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (single-node only in this script)",
    )
    parser.add_argument(
        "--max-tokens-per-rank",
        type=int,
        default=0,
        help="Max tokens per rank for buffer sizing (0 = num_tokens)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Print Kineto tables"
    )
    parser.add_argument(
        "--oneshot-ep",
        action="store_true",
        help="Use oneshot EP dispatch/combine",
    )
    ns = parser.parse_args()

    def _to_max_dtype(s: str) -> DType:
        mapping = {
            "bf16": DType.bfloat16,
            "fp16": DType.float16,
            "fp32": DType.float32,
        }
        return mapping[s]

    return EPBenchmarkArgs(
        num_tokens=ns.num_tokens,
        hidden=ns.hidden,
        num_topk=ns.num_topk,
        num_experts=ns.num_experts,
        dispatch_dtype=_to_max_dtype(ns.dispatch_dtype),
        combine_dtype=_to_max_dtype(ns.combine_dtype),
        iters=ns.iters,
        warmup=ns.warmup,
        gpus_per_node=ns.gpus_per_node,
        nodes=ns.nodes,
        max_tokens_per_rank=(ns.max_tokens_per_rank or None),
        profile=ns.profile,
        oneshot_ep=ns.oneshot_ep,
    )


def bench_ep(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    dispatch_dtype: torch.dtype,
    combine_dtype: torch.dtype,
    gpus_per_node: int = 0,
    nodes: int = 1,
    max_tokens_per_rank: int | None = None,
    iters: int = 30,
    profile: bool = False,
    oneshot_ep: bool = True,
) -> None:
    """
    Convenience API mirroring bench_blackwell_prefill.bench_prefill.
    Runs the MAX EP benchmark end-to-end and prints results.
    """
    args = EPBenchmarkArgs(
        num_tokens=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        num_experts=num_experts,
        dispatch_dtype=DType.from_torch(dispatch_dtype),
        combine_dtype=DType.from_torch(combine_dtype),
        iters=iters,
        warmup=0,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        max_tokens_per_rank=max_tokens_per_rank,
        profile=profile,
        oneshot_ep=oneshot_ep,
    )
    run_bench_max_ep(args)


# Default baseline run
if __name__ == "__main__":
    # NOTE: Running multiple tests sequentially fails due to NVSHMEM state not being
    # properly cleaned up between tests (CUDA_ERROR_ILLEGAL_ADDRESS on 2nd test).
    # This is a known issue: https://linear.app/modularml/issue/GENAI-361
    # To run other configurations, use CLI args:
    #   br //Kernels/benchmarks/comparison:bench_ep_baseline -- --num-tokens=256
    #   br //Kernels/benchmarks/comparison:bench_ep_baseline -- --hidden=8192
    bench_ep(128, 7168, 8, 64, torch.bfloat16, torch.bfloat16)
