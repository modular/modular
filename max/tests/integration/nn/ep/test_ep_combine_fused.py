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
"""The split and fused EP combine forms compute the same reduction.

``ep_combine_async`` + ``ep_combine_wait`` and the fused ``ep_combine`` run the
same collective over the same buffers and the same atomic counters: the split is
``combine_async_kernel`` followed by ``combine_wait``, the fused is
``combine_kernel``, which sends, waits and reduces in one launch.
``ExpertParallelMoE`` emits the fused form, so this pins that putting it where a
combine launch and its wait had nothing between them cannot change the result,
and that neither form leaves counter state a later call cannot consume.

``EP_COMBINE_AB=1`` additionally reports the per-call launch count and time of
both forms; that arm is a measurement, not a gate. Bazel does not forward the
client environment to the test action, so under Bazel the variable has to be
set for the action: ``./bazelw test
//max/tests/integration/nn/ep:test_ep_combine_fused
--test_filter=test_ep_combine_launch_cost_ab --test_env=EP_COMBINE_AB=1``.
Prefixing the command with ``EP_COMBINE_AB=1`` instead only sets it in the
Bazel client, and the arm silently skips.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pytest
import torch
from max import driver
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.ep_config import EPConfig
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer
from max.nn.legacy.comm.ep.ep_manager import get_ep_local_sync_counters_size


@dataclass
class CombinePair:
    """Both combine wirings over one EP buffer set, plus one input set."""

    split: Model
    """Compiled graph: split dispatch, then ``ep_combine_async`` + wait."""
    fused: Model
    """Compiled graph: split dispatch, then fused ``ep_combine``."""
    graph_inputs: list[Buffer]
    """The inputs both graphs execute on."""
    counters: list[Buffer]
    """The atomic counter buffers ``ep_init`` allocated."""
    devices: list[Accelerator]
    """The devices the graphs run on."""
    n_experts: int
    """Total experts across the ranks."""
    hidden_size: int
    """Model hidden dimension."""
    max_tokens_per_rank: int
    """Per-rank dispatch token capacity."""


def _setup(
    n_devices: int,
    max_tokens_per_rank: int,
    hidden_size: int = 512,
    top_k: int = 4,
    n_experts: int | None = None,
) -> CombinePair:
    """Compiles both combine wirings over one EP buffer set."""
    host = CPU(0)
    devices = [Accelerator(id) for id in range(n_devices)]
    session = InferenceSession(devices=[host, *devices])

    config = EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=hidden_size,
        top_k=top_k,
        n_experts=n_experts
        if n_experts is not None
        else min(64, n_devices * 4),
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
    )
    ep_initializer = EPCommInitializer(config)
    ep_initializer.ep_init(session)

    torch.manual_seed(42)
    device_ids = list(range(n_devices))

    input_lengths = torch.randint(1, config.max_tokens_per_rank, (n_devices,))
    per_device_inputs = [
        Buffer.from_dlpack(
            torch.randn(
                input_lengths[i],
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cpu",
            )
        ).to(devices[i])
        for i in range(n_devices)
    ]

    all_topk_ids = []
    for i in range(n_devices):
        expert_scores = torch.randn(input_lengths[i], config.n_experts)
        topk_ids = torch.topk(expert_scores, config.top_k, dim=1).indices.to(
            torch.int32
        )
        all_topk_ids.append(Buffer.from_dlpack(topk_ids).to(devices[i]))

    # Every routed expert contributes the same weight, so the reduction is a
    # plain sum and any accumulation-order difference between the two forms
    # shows up as a real mismatch instead of hiding in the weights.
    router_weights = [
        Buffer.from_dlpack(
            torch.ones(input_lengths[i], config.top_k, dtype=torch.float32)
        ).to(devices[i])
        for i in range(n_devices)
    ]

    per_device_input_types: list[TensorType | BufferType] = [
        TensorType(
            DType.bfloat16,
            (f"input_len_{i}", config.hidden_size),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]
    per_device_topk_ids_types: list[TensorType | BufferType] = [
        TensorType(
            DType.int32, (f"input_len_{i}", config.top_k), DeviceRef.GPU(i)
        )
        for i in range(n_devices)
    ]
    per_device_router_weight_types: list[TensorType | BufferType] = [
        TensorType(
            DType.float32, (f"input_len_{i}", config.top_k), DeviceRef.GPU(i)
        )
        for i in range(n_devices)
    ]

    ep_manager = EPBatchManager(config)

    def build_graph(fused_combine: bool) -> Graph:
        """Dispatch split, then combine in the form under test."""
        name = "ep_combine_fused" if fused_combine else "ep_combine_split"
        with Graph(
            name,
            input_types=per_device_input_types
            + per_device_topk_ids_types
            + per_device_router_weight_types
            + ep_manager.input_types(),
        ) as graph:
            n_inputs = n_devices * 3
            xs = [val.tensor for val in graph.inputs[:n_devices]]
            topk_ids = [
                val.tensor for val in graph.inputs[n_devices : n_devices * 2]
            ]
            weights = [
                val.tensor for val in graph.inputs[n_devices * 2 : n_inputs]
            ]

            ep_manager.fetch_buffers(graph.inputs[n_inputs:])

            dispatched: list[TensorValue] = []
            for i in device_ids:
                ep_manager.ep_dispatch_async(xs[i], topk_ids[i], i)
                dispatched.append(ep_manager.ep_dispatch_wait(i)[0])

            if fused_combine:
                combine_results = ep_manager.ep_combine_all(
                    dispatched, weights, device_ids
                )
            else:
                for i in device_ids:
                    ep_manager.ep_combine_async(dispatched[i], i)
                combine_results = [
                    ep_manager.ep_combine_wait(weights[i], i)
                    for i in device_ids
                ]

            graph.output(*combine_results)
            return graph

    return CombinePair(
        split=session.load(build_graph(fused_combine=False)),
        fused=session.load(build_graph(fused_combine=True)),
        graph_inputs=[
            *per_device_inputs,
            *all_topk_ids,
            *router_weights,
            *ep_initializer.model_inputs(),
        ],
        counters=ep_initializer.atomic_counters,
        devices=devices,
        n_experts=config.n_experts,
        hidden_size=config.hidden_size,
        max_tokens_per_rank=config.max_tokens_per_rank,
    )


def _to_host(value: Buffer) -> torch.Tensor:
    """Copies one device tensor to the host."""
    return torch.from_dlpack(value).cpu()


def _counter_state(counters: list[Buffer]) -> list[torch.Tensor]:
    """Snapshots the counters to the host, one host tensor each."""
    return [_to_host(counter).clone() for counter in counters]


MAX_GPUS_PER_NODE = 8
"""Guards the dispatch_wait counter layout, as ``test_ep_dispatch.py`` does."""


def _prefix_sum_words(n_experts: int) -> range:
    """The words holding dispatch_wait's within-expert rank prefix sums.

    ``EPLocalSyncCounters`` (``max/kernels/src/shmem/ep_comm.mojo``) lays the
    counters out as ``dispatch_async`` (``2 * n_experts + MAX_GPUS_PER_NODE``
    words), then ``dispatch_wait``'s region A (per expert-rank combine_async
    counts) and region B (the rank prefix sums) over ``n_experts`` words each.
    Region B is scratch the next dispatch rewrites and combine_async consumes,
    so its final value depends on which rank's block finished the update last
    and is not steady state either form can hold.
    """
    base = 2 * n_experts + MAX_GPUS_PER_NODE
    region = range(base + 2 * n_experts, base + 3 * n_experts)
    # Pinned against the size the Mojo struct gives the buffer, which
    # ``ep.init`` also asserts, so a layout change that moves or resizes this
    # window fails here instead of silently widening what the test tolerates.
    assert region.stop <= get_ep_local_sync_counters_size(n_experts), (
        f"the dispatch_wait prefix-sum window {region} runs past the"
        f" {get_ep_local_sync_counters_size(n_experts)}-word counter buffer"
        " EPLocalSyncCounters sizes; the layout moved"
    )
    return region


def _assert_same_counters(
    got: list[torch.Tensor],
    want: list[torch.Tensor],
    when: str,
    scratch: range = range(0),
) -> None:
    """Asserts two counter snapshots agree, outside ``scratch``."""
    assert len(got) == len(want), when
    for idx, (a, b) in enumerate(zip(got, want, strict=True)):
        if torch.equal(a, b):
            continue
        differing = [
            int(i)
            for i in (a != b).nonzero().flatten().tolist()
            if int(i) not in scratch
        ]
        if not differing:
            continue
        detail = ", ".join(
            f"[{i}] {a[i].item()} -> {b[i].item()}" for i in differing[:8]
        )
        raise AssertionError(
            f"{when}: counter buffer {idx} differs in {len(differing)} of"
            f" {a.numel()} words: {detail}"
        )


@pytest.mark.parametrize("n_devices", [4])
def test_ep_combine_fused_matches_split(n_devices: int) -> None:
    """Fused and split combine agree, and neither leaves stale counters."""
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )
    pair = _setup(n_devices, max_tokens_per_rank=64)

    def run(model: Model) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        outputs = [
            _to_host(value).clone()
            for value in model.execute(*pair.graph_inputs)
        ]
        return outputs, _counter_state(pair.counters)

    # The first call after ``ep_init`` runs from an all-zero counter state the
    # steady state never revisits: it leaves ``dispatch_wait``'s prefix-sum
    # scratch word at 0 where a warm call leaves the rank count. That word is
    # dispatch residue the next dispatch overwrites, not combine protocol
    # state, so each form warms up before its counter state is compared. What
    # must hold is that the steady state repeats, which is what a decode loop
    # relies on.
    run(pair.split)
    split_out_1, split_counters_1 = run(pair.split)
    split_out_2, split_counters_2 = run(pair.split)
    scratch = _prefix_sum_words(pair.n_experts)
    _assert_same_counters(
        split_counters_1, split_counters_2, "split repeat", scratch
    )

    run(pair.fused)
    fused_out_1, fused_counters_1 = run(pair.fused)
    fused_out_2, fused_counters_2 = run(pair.fused)
    _assert_same_counters(
        fused_counters_1, fused_counters_2, "fused repeat", scratch
    )
    # The two forms must also agree with each other, word for word outside the
    # scratch window: a form that left extra state behind would show here.
    _assert_same_counters(
        fused_counters_1, split_counters_1, "split vs fused", scratch
    )

    for device_idx in range(n_devices):
        split_result = split_out_1[device_idx]
        fused_result = fused_out_1[device_idx]
        assert torch.equal(split_result, fused_result), (
            f"device {device_idx}: fused combine differs from the split pair;"
            f" max |diff| ="
            f" {(split_result.float() - fused_result.float()).abs().max().item()}"
        )
        for label, second in (
            ("split", split_out_2[device_idx]),
            ("fused", fused_out_2[device_idx]),
        ):
            got = split_result if label == "split" else fused_result
            assert torch.equal(got, second), (
                f"{label} device {device_idx}: repeat run changed the output"
            )


@pytest.mark.skipif(
    os.environ.get("EP_COMBINE_AB") != "1",
    reason=(
        "Set EP_COMBINE_AB=1 to measure; this arm reports, it does not gate."
        " Under Bazel, pass --test_env=EP_COMBINE_AB=1: the client"
        " environment does not reach the test action."
    ),
)
def test_ep_combine_launch_cost_ab() -> None:
    """Reports launches per call and per-call time for both combine forms."""
    n_devices = min(accelerator_count(), int(os.environ.get("EP_AB_GPUS", 8)))
    assert n_devices >= 2, "EP needs at least two devices"
    pair = _setup(
        n_devices,
        max_tokens_per_rank=int(os.environ.get("EP_AB_TOKENS", 128)),
        hidden_size=int(os.environ.get("EP_AB_HIDDEN", 7168)),
        top_k=int(os.environ.get("EP_AB_TOP_K", 8)),
        n_experts=int(os.environ.get("EP_AB_EXPERTS", 128)),
    )

    def synchronize() -> None:
        for device in pair.devices:
            device.synchronize()

    arms = {"split": pair.split, "fused": pair.fused}
    wanted = os.environ.get("EP_COMBINE_ARM", "both")
    if wanted != "both":
        arms = {wanted: arms[wanted]}

    iterations = int(os.environ.get("EP_AB_ITERS", 50))
    for label, model in arms.items():
        for _ in range(10):
            model.execute(*pair.graph_inputs)
        synchronize()

        with driver.launch_trace() as entries:
            model.execute(*pair.graph_inputs)
        synchronize()
        combine_launches = sum(
            1 for entry in entries if "combine" in entry.name
        )

        start = time.perf_counter()
        for _ in range(iterations):
            model.execute(*pair.graph_inputs)
            synchronize()
        per_call = (time.perf_counter() - start) / iterations * 1e6

        print(
            f"AB {label}: devices={n_devices}"
            f" tokens_per_rank={pair.max_tokens_per_rank}"
            f" hidden={pair.hidden_size} experts={pair.n_experts}"
            f" launches/call={len(entries)}"
            f" combine_launches/call={combine_launches}"
            f" us/call={per_call:.1f}"
        )
