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
# Run via Bazel: br //max/kernels/benchmarks/misc/comparison:bench_ep_baseline
#
# This script establishes baseline performance metrics for MAX EP
# dispatch + grouped_matmul + combine operations.
# Supports two modes:
#   1. Separate kernels (default): Times dispatch_async, dispatch_wait,
#      combine_async, combine_wait
#   2. Fused kernels (--oneshot-ep): Times fused dispatch and combine kernels
# Reports effective GB/s for each phase.
#
# Dispatch dtype options:
#   --dispatch-dtype bf16   (default)
#   --dispatch-dtype fp8    (blockwise FP8 quantized dispatch)
#   --dispatch-dtype nvfp4  (NVFP4 quantized dispatch)

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
from max.experimental.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Allreduce, Signals
from max.nn.comm.ep.ep_config import EPConfig
from max.nn.comm.ep.ep_manager import EPBatchManager, EPCommInitializer
from max.nn.kernels import (
    grouped_dynamic_scaled_fp8_matmul,
    grouped_dynamic_scaled_nvfp4_matmul,
    grouped_matmul_ragged,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)


def _ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _make_dispatch_quant_config(dispatch_dtype: DType) -> QuantConfig | None:
    """Build the QuantConfig needed by EPConfig for quantized dispatch."""
    if dispatch_dtype.is_float8():
        return QuantConfig(
            input_scale=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, 128),
            ),
            weight_scale=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(128, 128),
            ),
            mlp_quantized_layers=set(),
            attn_quantized_layers=set(),
            format=QuantFormat.BLOCKSCALED_FP8,
        )
    if dispatch_dtype == DType.uint8:
        return QuantConfig(
            input_scale=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float8_e4m3fn,
                block_size=(1, 16),
            ),
            weight_scale=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float8_e4m3fn,
                block_size=(128, 128),
            ),
            mlp_quantized_layers=set(),
            attn_quantized_layers=set(),
            format=QuantFormat.NVFP4,
        )
    return None


def _call_grouped_matmul(
    dispatched: tuple[TensorValue, ...],
    weight: TensorValue,
    config: EPConfig,
    weight_scales: TensorValue | None = None,
    expert_scales: TensorValue | None = None,
) -> TensorValue:
    """Invoke the grouped matmul that matches the dispatch format.

    The dispatch output tuple layout varies by format:
      BF16:  (tokens, expert_start, expert_ids, usage_stats)
      FP8:   (tokens, a_scales, expert_start, expert_ids, usage_stats)
      NVFP4: (tokens, a_scales, expert_start, a_scale_offsets,
              expert_ids, usage_stats)
    """
    qc = config.dispatch_quant_config
    if qc is not None and qc.is_nvfp4:
        (
            tokens,
            a_scales,
            expert_start,
            a_scale_offsets,
            expert_ids,
            usage_stats,
        ) = dispatched
        assert weight_scales is not None and expert_scales is not None
        return grouped_dynamic_scaled_nvfp4_matmul(
            tokens,
            weight,
            a_scales,
            weight_scales,
            expert_start,
            a_scale_offsets,
            expert_ids,
            expert_scales.to(tokens.device),
            usage_stats,
        )
    elif config.dispatch_dtype.is_float8():
        tokens, a_scales, expert_start, expert_ids, usage_stats = dispatched
        assert qc is not None and weight_scales is not None
        return grouped_dynamic_scaled_fp8_matmul(
            tokens,
            weight,
            a_scales,
            weight_scales,
            expert_start,
            expert_ids,
            usage_stats,
            qc.input_scale,
            qc.weight_scale,
            tokens_padded_per_expert=True,
        )
    else:
        tokens, expert_start, expert_ids, usage_stats = dispatched
        return grouped_matmul_ragged(
            tokens,
            weight,
            expert_start,
            expert_ids,
            usage_stats,
        )


def _matmul_input_types(
    config: EPConfig,
    n_local_experts: int,
) -> list[TensorType]:
    """Build graph-input TensorTypes for the grouped-matmul weights/scales.

    The returned list is ordered as:
      [weight_gpu0, ..., weight_gpuN,
       b_scales_gpu0, ..., b_scales_gpuN,   (fp8 / nvfp4 only)
       expert_scales_gpu0, ..., expert_scales_gpuN]  (nvfp4 only)
    """
    types: list[TensorType] = []
    hidden = config.hidden_size
    qc = config.dispatch_quant_config
    is_nvfp4 = qc is not None and qc.is_nvfp4
    is_fp8 = config.dispatch_dtype.is_float8()

    for dev_id in range(config.n_gpus_per_node):
        dev = DeviceRef.GPU(dev_id)
        if is_nvfp4:
            types.append(
                TensorType(
                    DType.uint8,
                    [n_local_experts, hidden, hidden // 2],
                    device=dev,
                )
            )
        elif is_fp8:
            types.append(
                TensorType(
                    DType.float8_e4m3fn,
                    [n_local_experts, hidden, hidden],
                    device=dev,
                )
            )
        else:
            types.append(
                TensorType(
                    DType.bfloat16,
                    [n_local_experts, hidden, hidden],
                    device=dev,
                )
            )

    if is_fp8:
        scale_n = _ceildiv(hidden, 128)
        scale_k = _ceildiv(hidden, 128)
        for dev_id in range(config.n_gpus_per_node):
            types.append(
                TensorType(
                    DType.float32,
                    [n_local_experts, scale_n, scale_k],
                    device=DeviceRef.GPU(dev_id),
                )
            )
    elif is_nvfp4:
        scale_n = _ceildiv(hidden, 128)
        scale_k = _ceildiv(hidden, 64)
        for dev_id in range(config.n_gpus_per_node):
            types.append(
                TensorType(
                    DType.float8_e4m3fn,
                    [n_local_experts, scale_n, scale_k, 32, 4, 4],
                    device=DeviceRef.GPU(dev_id),
                )
            )

    if is_nvfp4:
        for dev_id in range(config.n_gpus_per_node):
            types.append(
                TensorType(
                    DType.float32,
                    [n_local_experts],
                    device=DeviceRef.GPU(dev_id),
                )
            )

    return types


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
    max_tokens_per_rank: int
    profile: bool
    oneshot_ep: bool = False
    fused_shared_expert: bool = False


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
    """
    manager = EPBatchManager(config)
    n = config.n_gpus_per_node
    n_ranks = config.n_gpus_per_node * config.n_nodes
    n_local_experts = config.n_experts // n_ranks
    n_matmul_experts = n_local_experts + (
        1 if config.fused_shared_expert else 0
    )
    qc = config.dispatch_quant_config
    is_nvfp4 = qc is not None and qc.is_nvfp4
    is_fp8 = config.dispatch_dtype.is_float8()
    has_weight_scales = is_fp8 or is_nvfp4

    ep_static_input_types = list(manager.input_types())

    # Input tokens are always bf16 -- the dispatch kernel handles quantization
    input_dtype = (
        DType.bfloat16 if (is_fp8 or is_nvfp4) else config.dispatch_dtype
    )

    token_types: list[TensorType] = []
    topk_types: list[TensorType] = []
    router_weight_types: list[TensorType] = []
    for dev_id in range(n):
        token_types.append(
            TensorType(
                dtype=input_dtype,
                shape=["num_tokens", config.hidden_size],
                device=DeviceRef.GPU(dev_id),
            )
        )
        topk_types.append(
            TensorType(
                dtype=DType.int32,
                shape=["num_tokens", config.top_k],
                device=DeviceRef.GPU(dev_id),
            )
        )
        router_weight_types.append(
            TensorType(
                dtype=DType.float32,
                shape=["num_tokens", config.top_k],
                device=DeviceRef.GPU(dev_id),
            )
        )

    matmul_types = _matmul_input_types(config, n_matmul_experts)

    # Allreduce signal buffers for synchronizing kernel launches across GPUs
    signals = Signals(devices=[DeviceRef.GPU(dev_id) for dev_id in range(n)])
    allreduce = Allreduce(num_accelerators=n)
    signal_input_types = list(signals.input_types())

    with Graph(
        "ep_bench",
        input_types=[
            *ep_static_input_types,
            *token_types,
            *topk_types,
            *router_weight_types,
            *matmul_types,
            *signal_input_types,
        ],
    ) as g:
        total_static = len(ep_static_input_types)
        manager.fetch_buffers(g.inputs[:total_static])

        # Unpack per-device dynamic inputs
        off = total_static
        tokens_vals = [g.inputs[off + i].tensor for i in range(n)]
        off += n
        topk_vals = [g.inputs[off + i].tensor for i in range(n)]
        off += n
        router_weights_vals = [g.inputs[off + i].tensor for i in range(n)]
        off += n

        # Unpack matmul weight / scale inputs
        weight_vals = [g.inputs[off + i].tensor for i in range(n)]
        off += n
        weight_scale_vals: list[TensorValue | None] = [None] * n
        expert_scale_vals: list[TensorValue | None] = [None] * n
        if has_weight_scales:
            weight_scale_vals = [g.inputs[off + i].tensor for i in range(n)]
            off += n
        if is_nvfp4:
            expert_scale_vals = [g.inputs[off + i].tensor for i in range(n)]
            off += n

        # Unpack allreduce signal buffers
        signal_bufs = [g.inputs[off + i].buffer for i in range(n)]
        off += n
        assert off == len(g.inputs)

        # Allreduce on token inputs as a barrier so all GPUs launch EP
        # dispatch at roughly the same time (improves timing accuracy).
        tokens_vals = allreduce(tokens_vals, signal_bufs)

        # Per-device NVFP4 input_scales constant for dispatch quantization
        nvfp4_input_scales: list[TensorValue | None] = [None] * n
        if is_nvfp4:
            for dev_id in range(n):
                nvfp4_input_scales[dev_id] = ops.constant(
                    [1.0] * config.n_experts,
                    dtype=DType.float32,
                    device=DeviceRef.GPU(dev_id),
                )

        # -- Dispatch --
        dispatched: list[tuple[TensorValue, ...]] = []
        if oneshot_ep:
            for dev_id in range(n):
                dispatched.append(
                    manager.ep_dispatch(
                        tokens_vals[dev_id],
                        topk_vals[dev_id],
                        device_id=dev_id,
                        input_scales=nvfp4_input_scales[dev_id],
                    )
                )
        else:
            for dev_id in range(n):
                manager.ep_dispatch_async(
                    tokens_vals[dev_id],
                    topk_vals[dev_id],
                    device_id=dev_id,
                    input_scales=nvfp4_input_scales[dev_id],
                )
            for dev_id in range(n):
                dispatched.append(manager.ep_dispatch_wait(device_id=dev_id))

        # -- Grouped matmul (quantized -> bf16) --
        matmul_outs: list[TensorValue] = []
        for dev_id in range(n):
            matmul_outs.append(
                _call_grouped_matmul(
                    dispatched[dev_id],
                    weight_vals[dev_id],
                    config,
                    weight_scales=weight_scale_vals[dev_id],
                    expert_scales=expert_scale_vals[dev_id],
                )
            )

        # Allreduce on matmul outputs as a barrier so all GPUs launch EP combine
        # at roughly the same time (improves timing accuracy).
        matmul_outs = allreduce(matmul_outs, signal_bufs)

        # -- Combine --
        outputs: list[TensorValue] = []
        if oneshot_ep:
            for dev_id in range(n):
                out = manager.ep_combine(
                    matmul_outs[dev_id],
                    router_weights_vals[dev_id],
                    device_id=dev_id,
                )
                outputs.append(ops.sum(out))
        else:
            for dev_id in range(n):
                manager.ep_combine_async(matmul_outs[dev_id], device_id=dev_id)
            for dev_id in range(n):
                out = manager.ep_combine_wait(
                    router_weights_vals[dev_id], device_id=dev_id
                )
                outputs.append(ops.sum(out))

        # simulate the residual connection that would be fused into the EP
        # combine kernel.
        outputs = [
            inp + out for inp, out in zip(tokens_vals, outputs, strict=True)
        ]

        g.output(*outputs)
        return g


def make_inputs_for_execute(
    config: EPConfig,
    initializer: EPCommInitializer,
    num_tokens: int,
    n_local_experts: int,
    signal_buffers: list[Buffer],
) -> list[Buffer]:
    """Prepare input Buffers for model.execute().

    Order matches the graph input_types in build_ep_graph:
      EP static | tokens | topk | router_weights | matmul weight/scales
      | signal_buffers
    """
    inputs: list[Buffer] = []
    # Static EP inputs
    inputs.extend(initializer.model_inputs())

    hidden = config.hidden_size
    qc = config.dispatch_quant_config
    is_nvfp4 = qc is not None and qc.is_nvfp4
    is_fp8 = config.dispatch_dtype.is_float8()

    # Tokens are always bf16 (dispatch kernel quantizes internally)
    for dev_id in range(config.n_gpus_per_node):
        with torch.cuda.device(dev_id):
            x = torch.randn(
                (num_tokens, hidden),
                dtype=torch.bfloat16,
                device=f"cuda:{dev_id}",
            )
            inputs.append(Buffer.from_dlpack(x))

    # Then append all topk tensors
    for dev_id in range(config.n_gpus_per_node):
        with torch.cuda.device(dev_id):
            # Sample top_k unique expert IDs per token via argsort of random scores
            topk = torch.argsort(
                torch.rand(
                    num_tokens, config.n_experts, device=f"cuda:{dev_id}"
                ),
                dim=1,
            )[:, : config.top_k].to(torch.int32)
            inputs.append(Buffer.from_dlpack(topk))

    # Then append router weights (uniform weights for benchmark)
    for dev_id in range(config.n_gpus_per_node):
        with torch.cuda.device(dev_id):
            rw = (
                torch.ones(
                    (num_tokens, config.top_k),
                    dtype=torch.float32,
                    device=f"cuda:{dev_id}",
                )
                / config.top_k
            )
            inputs.append(Buffer.from_dlpack(rw))

    # -- Matmul weights (one copy per device) --
    for dev_id in range(config.n_gpus_per_node):
        with torch.cuda.device(dev_id):
            dev = f"cuda:{dev_id}"
            if is_nvfp4:
                w = torch.randint(
                    0,
                    255,
                    (n_local_experts, hidden, hidden // 2),
                    dtype=torch.uint8,
                    device=dev,
                )
            elif is_fp8:
                w = torch.randn(n_local_experts, hidden, hidden, device=dev).to(
                    torch.float8_e4m3fn
                )
            else:
                w = torch.randn(
                    n_local_experts,
                    hidden,
                    hidden,
                    dtype=torch.bfloat16,
                    device=dev,
                )
            inputs.append(Buffer.from_dlpack(w))

    # -- b_scales --
    if is_fp8:
        scale_n = _ceildiv(hidden, 128)
        scale_k = _ceildiv(hidden, 128)
        for dev_id in range(config.n_gpus_per_node):
            with torch.cuda.device(dev_id):
                bs = torch.ones(
                    n_local_experts,
                    scale_n,
                    scale_k,
                    dtype=torch.float32,
                    device=f"cuda:{dev_id}",
                )
                inputs.append(Buffer.from_dlpack(bs))
    elif is_nvfp4:
        scale_n = _ceildiv(hidden, 128)
        scale_k = _ceildiv(hidden, 64)
        for dev_id in range(config.n_gpus_per_node):
            with torch.cuda.device(dev_id):
                bs = torch.ones(
                    n_local_experts,
                    scale_n,
                    scale_k,
                    32,
                    4,
                    4,
                    dtype=torch.bfloat16,
                    device=f"cuda:{dev_id}",
                ).to(torch.float8_e4m3fn)
                inputs.append(Buffer.from_dlpack(bs))

    # -- expert_scales (nvfp4 only) --
    if is_nvfp4:
        for dev_id in range(config.n_gpus_per_node):
            with torch.cuda.device(dev_id):
                es = torch.ones(
                    n_local_experts,
                    dtype=torch.float32,
                    device=f"cuda:{dev_id}",
                )
                inputs.append(Buffer.from_dlpack(es))

    # -- allreduce signal buffers --
    inputs.extend(signal_buffers)

    return inputs


def compute_bytes_per_token(
    hidden: int,
    dtype: DType,
    fp8_scale_bytes: int = 4,
) -> float:
    """Approximate bytes per token transferred.

    - BF16: hidden * 2
    - FP8 blockwise (1x128): hidden + (hidden / 128) * fp8_scale_bytes
    - NVFP4: hidden/2 + hidden/16  (packed uint8 + fp8 scales)
    """
    if dtype == DType.uint8:
        return float(hidden // 2) + float(hidden // 16)
    if dtype.is_float8():
        return float(hidden) + float(hidden // 128) * fp8_scale_bytes
    if dtype == DType.bfloat16:
        return float(hidden * 2)
    raise ValueError(f"Unsupported dtype for bytes-per-token: {dtype}")


def run_bench_max_ep(args: EPBenchmarkArgs) -> None:
    # Determine GPU topology
    visible_gpus = torch.cuda.device_count()
    assert visible_gpus >= 1, "CUDA device not found"
    n_gpus = args.gpus_per_node or visible_gpus
    assert n_gpus <= visible_gpus, (
        f"Requested {n_gpus} GPUs, but only {visible_gpus} visible"
    )

    # Prepare EP config
    assert args.num_tokens > 0, "num_tokens must be greater than 0"
    assert args.num_tokens <= args.max_tokens_per_rank, (
        "num_tokens must be less than or equal to max_tokens_per_rank"
    )
    dispatch_quant_config = _make_dispatch_quant_config(args.dispatch_dtype)
    config = EPConfig(
        dispatch_dtype=args.dispatch_dtype,
        combine_dtype=args.combine_dtype,
        hidden_size=args.hidden,
        top_k=args.num_topk,
        n_experts=args.num_experts,
        max_tokens_per_rank=args.max_tokens_per_rank,
        n_gpus_per_node=n_gpus,
        n_nodes=args.nodes,
        dispatch_quant_config=dispatch_quant_config,
        fused_shared_expert=args.fused_shared_expert,
    )

    n_ranks = n_gpus * args.nodes
    n_local_experts = args.num_experts // n_ranks
    n_matmul_experts = n_local_experts + (1 if args.fused_shared_expert else 0)

    # Session with all local GPUs
    session = InferenceSession(devices=[Accelerator(i) for i in range(n_gpus)])

    # Initialize NVSHMEM buffers
    initializer = EPCommInitializer(config)
    initializer.ep_init(session)

    # Build and compile EP bench graph
    graph = build_ep_graph(config, oneshot_ep=args.oneshot_ep)
    model = session.load(graph)

    # Prepare runtime inputs
    signals = Signals(devices=[DeviceRef.GPU(i) for i in range(n_gpus)])
    execute_inputs = make_inputs_for_execute(
        config,
        initializer,
        args.num_tokens,
        n_matmul_experts,
        signals.buffers(),
    )

    def run_once() -> list[Buffer]:
        return model.execute(*execute_inputs)

    # Compute effective bandwidth per device (bytes / avg_time)
    bytes_per_token_dispatch = compute_bytes_per_token(
        args.hidden, args.dispatch_dtype
    )
    bytes_per_token_combine = compute_bytes_per_token(
        args.hidden, args.combine_dtype
    )
    dispatch_bytes = args.num_tokens * args.num_topk * bytes_per_token_dispatch
    combine_bytes = args.num_tokens * args.num_topk * bytes_per_token_combine

    print("=" * 80)
    print(
        f"MAX EP Benchmark (tokens={args.num_tokens}, hidden={args.hidden}, "
        f"top_k={args.num_topk}, experts={args.num_experts}, gpus={n_gpus})"
    )
    print(f"Dispatch dtype: {args.dispatch_dtype}")
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
        choices=["bf16", "fp8", "nvfp4"],
        help="Dispatch activation dtype",
    )
    parser.add_argument(
        "--combine-dtype",
        type=str,
        default="bf16",
        choices=["bf16"],
        help="Combine activation dtype (only bf16 supported)",
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
        default=128,
        help="Max tokens per rank for buffer sizing",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Print Kineto tables"
    )
    parser.add_argument(
        "--oneshot-ep",
        action="store_true",
        help="Use oneshot EP dispatch/combine",
    )
    parser.add_argument(
        "--fused-shared-expert",
        action="store_true",
        help="Fuse shared expert into routed expert list (+1 local expert)",
    )
    ns = parser.parse_args()

    def _to_max_dtype(s: str) -> DType:
        mapping: dict[str, DType] = {
            "bf16": DType.bfloat16,
            "fp8": DType.float8_e4m3fn,
            "nvfp4": DType.uint8,
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
        max_tokens_per_rank=ns.max_tokens_per_rank,
        profile=ns.profile,
        oneshot_ep=ns.oneshot_ep,
        fused_shared_expert=ns.fused_shared_expert,
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
    max_tokens_per_rank: int = 128,
    iters: int = 30,
    profile: bool = False,
    oneshot_ep: bool = True,
    fused_shared_expert: bool = False,
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
        dispatch_dtype=torch_dtype_to_max(dispatch_dtype),
        combine_dtype=torch_dtype_to_max(combine_dtype),
        iters=iters,
        warmup=0,
        gpus_per_node=gpus_per_node,
        nodes=nodes,
        max_tokens_per_rank=max_tokens_per_rank,
        profile=profile,
        oneshot_ep=oneshot_ep,
        fused_shared_expert=fused_shared_expert,
    )
    run_bench_max_ep(args)


# Default baseline run
if __name__ == "__main__":
    # NOTE: Running multiple tests sequentially fails due to NVSHMEM state not
    # being properly cleaned up between tests (CUDA_ERROR_ILLEGAL_ADDRESS on
    # 2nd test).  This is a known issue:
    # https://linear.app/modularml/issue/GENAI-361
    #
    # Examples:
    #   br //max/kernels/benchmarks/misc/comparison:bench_ep_baseline -- \
    #       --oneshot-ep
    #
    # Kimi-K2.5 config:
    #   br //max/kernels/benchmarks/misc/comparison:bench_ep_baseline -- \
    #       --oneshot-ep --dispatch-dtype=nvfp4 --fused-shared-expert \
    #       --iters 300 --num-tokens 1 --num-experts 384 --gpus-per-node 8
    #
    # DeepSeek-V3/R1 config:
    #   br //max/kernels/benchmarks/misc/comparison:bench_ep_baseline -- \
    #       --oneshot-ep --dispatch-dtype=nvfp4 --fused-shared-expert \
    #       --iters 300 --num-tokens 1 --num-experts 256 --gpus-per-node 8
    run_bench_max_ep(parse_args())
