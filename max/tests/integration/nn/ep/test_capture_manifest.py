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

"""Checks a captured device graph's profiler manifest against the graph itself.

The manifest is what a profiler resolves replayed kernels through: each
operation MAX enqueued during the capture, with the op span that enqueued it.
The driver's own record of the graph is the verbose DOT dump MLRT writes under
device-context.log_graph_capture, where the manifest is written too.
Every kernel node MAX launched must appear in the manifest with an op label —
including kernels on streams forked into the capture, such as the EP
shared-expert branch — and the only kernel nodes allowed to be missing are
vendor-library launches, which never pass MAX's launch site.

Needs no profiler, so it runs on CI executors without a device-trace backend.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_api,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType, ops
from max.nn import Allreduce, Signals
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.moe import MoE
from max.nn.moe.expert_parallel import forward_moe_sharded_layers

N_DEVICES = 2
# Per test run, so concurrent runs cannot see each other's dumps. The device
# contexts read the setting when they are created, so it must be in the
# environment before the first Accelerator.
LOG_DIR = Path(os.environ.get("TEST_TMPDIR", "/tmp")) / "capture_manifest"
os.environ["MODULAR_DEVICE_CONTEXT_LOG_GRAPH_CAPTURE"] = str(LOG_DIR)

# Kernel symbols of the vendor libraries a MAX graph can call into: cuBLAS /
# cuBLASLt, NCCL / RCCL, hipBLASLt (Tensile).
_VENDOR_KERNEL = re.compile(r"^(nvjet_|_ZN8cublasLt|.*nccl|Cijk_)")

_NODE_RE = re.compile(r'"(\w+)"\[[^\n]*?label="\{(.*?)\}"\];', re.DOTALL)
_KERNEL_NAME_RE = re.compile(
    r"ID \| \d+(?: \(topoId: \d+\))? \| (.*?)\\<\\<\\<", re.DOTALL
)


def _dot_kernels(path: Path) -> list[str]:
    names = []
    for match in _NODE_RE.finditer(path.read_text(errors="replace")):
        body = match.group(2).strip()
        if body.split("\n", 1)[0].strip() == "KERNEL":
            name = _KERNEL_NAME_RE.search(body)
            names.append(name.group(1).strip() if name else "")
    return names


def _snapshot() -> tuple[set[Path], set[Path]]:
    return set(LOG_DIR.glob("*.dot")), set(
        LOG_DIR.glob("graph_manifest_*.json")
    )


def _check_capture(
    case: str, before: tuple[set[Path], set[Path]]
) -> list[dict[str, object]]:
    """Asserts the invariant for every graph a capture produced."""
    dots_now, manifests_now = _snapshot()

    def order(p: Path) -> int:
        return int(re.findall(r"\d+", p.name)[-1])

    dots = sorted(dots_now - before[0], key=order)
    manifests = [
        json.loads(p.read_text())
        for p in sorted(manifests_now - before[1], key=order)
    ]
    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")) / case
    out.mkdir(parents=True, exist_ok=True)
    for path in [*dots, *sorted(manifests_now - before[1])]:
        shutil.copy(path, out / path.name)

    assert len(dots) == N_DEVICES and len(manifests) == N_DEVICES, (
        f"expected one DOT dump and one manifest per device, got "
        f"{len(dots)} and {len(manifests)}"
    )
    for dot, manifest in zip(dots, manifests, strict=True):
        strings = manifest["strings"]
        all_entries = manifest["entries"]
        assert isinstance(strings, list) and isinstance(all_entries, list)
        entries = [e for e in all_entries if e[0] == "k"]
        recorded = Counter(strings[e[1]] for e in entries)
        unlabeled = [strings[e[1]] for e in entries if not strings[e[2]]]
        assert not unlabeled, (
            f"{case}: kernels without an op label: {unlabeled}"
        )

        missing = Counter(_dot_kernels(dot)) - recorded
        not_vendor = [
            n for n in missing.elements() if not _VENDOR_KERNEL.match(n)
        ]
        print(
            f"{case} {dot.name}: {sum(recorded.values())} recorded,"
            f" {sum(missing.values())} vendor kernels unrecorded"
        )
        assert not not_vendor, (
            f"{case}: MAX kernel nodes absent from the manifest: {not_vendor}"
        )
    return manifests


def _setup() -> None:
    if accelerator_count() < N_DEVICES:
        pytest.skip("needs 2 GPUs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize(
    "vendor_blas,vendor_ccl",
    [("false", "false"), ("true", "false"), ("false", "true")],
)
def test_tp_manifest_matches_graph(vendor_blas: str, vendor_ccl: str) -> None:
    _setup()
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    refs = [DeviceRef.GPU(i) for i in range(N_DEVICES)]
    signals = Signals(devices=refs)
    side = 256
    types = [TensorType(DType.bfloat16, [side, side], r) for r in refs]
    with Graph(
        "tp_capture_manifest",
        input_types=[*types, *types, *signals.input_types()],
    ) as graph:
        xs = [v.tensor for v in graph.inputs[:N_DEVICES]]
        ws = [v.tensor for v in graph.inputs[N_DEVICES : 2 * N_DEVICES]]
        partial = [ops.matmul(x, w) for x, w in zip(xs, ws, strict=True)]
        reduced = Allreduce(num_accelerators=N_DEVICES)(
            partial, [v.buffer for v in graph.inputs[2 * N_DEVICES :]]
        )
        graph.output(*[ops.silu(r) for r in reduced])

    session = InferenceSession(devices=[CPU(), *devices])
    session._use_vendor_blas(vendor_blas)
    session._use_vendor_ccl(vendor_ccl)
    model = session.load(graph)
    rng = np.random.default_rng(0)
    args = [
        Buffer.from_dlpack(
            torch.from_numpy(
                rng.standard_normal((side, side), dtype=np.float32)
            ).to(torch.bfloat16)
        ).to(devices[i % N_DEVICES])
        for i in range(2 * N_DEVICES)
    ]
    signal_buffers = signals.buffers()
    model.execute(*args, *signal_buffers)
    before = _snapshot()
    captured = model.capture(11, *args, *signal_buffers)
    model.replay(11, *args, *signal_buffers)
    _check_capture(f"tp_blas-{vendor_blas}_ccl-{vendor_ccl}", before)
    del captured


def test_ep_manifest_records_forked_stream() -> None:
    _setup()
    if accelerator_api() == "hip":
        pytest.skip("EP graph capture fails on AMD (KERN-3621)")
    hidden, moe_dim, n_experts, top_k = 2048, 1024, 16, 4
    dtype = DType.bfloat16
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    refs = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)
    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=dtype,
        hidden_size=hidden,
        top_k=top_k,
        n_experts=n_experts,
        max_tokens_per_rank=64,
        n_gpus_per_node=N_DEVICES,
        n_nodes=1,
    )
    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)
    moe = MoE(
        devices=refs,
        hidden_dim=hidden,
        num_experts=n_experts,
        num_experts_per_token=top_k,
        moe_dim=moe_dim,
        has_shared_experts=True,
        shared_experts_dim=moe_dim,
        ep_size=N_DEVICES,
        dtype=dtype,
        ep_batch_manager=ep_batch_manager,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(N_DEVICES)
    shards = moe.shard(refs)
    torch.manual_seed(0)
    weights = {"gate.gate_score.weight": torch.randn(n_experts, hidden) * 0.01}
    for e in range(n_experts):
        weights[f"experts.{e}.gate_proj.weight"] = (
            torch.randn(moe_dim, hidden) * 0.01
        )
        weights[f"experts.{e}.up_proj.weight"] = (
            torch.randn(moe_dim, hidden) * 0.01
        )
        weights[f"experts.{e}.down_proj.weight"] = (
            torch.randn(hidden, moe_dim) * 0.01
        )
    for p in ("gate", "up"):
        weights[f"shared_experts.{p}_proj.weight"] = (
            torch.randn(moe_dim, hidden) * 0.01
        )
    weights["shared_experts.down_proj.weight"] = (
        torch.randn(hidden, moe_dim) * 0.01
    )
    moe.load_state_dict({k: v.to(torch.bfloat16) for k, v in weights.items()})
    ep_comm_init.ep_init(session)

    in_types = [
        TensorType(dtype, [32, hidden], DeviceRef.GPU(i))
        for i in range(N_DEVICES)
    ]
    with Graph(
        "ep_capture_manifest",
        input_types=[*in_types, *ep_batch_manager.input_types()],
    ) as graph:
        xs = [v.tensor for v in graph.inputs[:N_DEVICES]]
        ep_batch_manager.fetch_buffers(graph.inputs[N_DEVICES:])
        graph.output(*forward_moe_sharded_layers(shards, xs))
    model = session.load(graph, weights_registry=moe.state_dict())
    inputs = [
        Buffer.from_dlpack(torch.randn(32, hidden, dtype=torch.bfloat16)).to(d)
        for d in devices
    ]
    model.execute(*inputs, *ep_comm_init.model_inputs())
    before = _snapshot()
    captured = model.capture(12, *inputs, *ep_comm_init.model_inputs())
    model.replay(12, *inputs, *ep_comm_init.model_inputs())
    manifests = _check_capture("ep", before)
    # Replay verification compares an eager run against the capture stream's
    # own operations, so the forked branch recorded above must not fail it.
    model.debug_verify_replay(12, *inputs, *ep_comm_init.model_inputs())
    del captured

    # The shared-expert branch runs on a stream forked into the capture; its
    # kernels land in the owning graph's manifest under a nonzero stream index.
    for manifest in manifests:
        entries = manifest["entries"]
        assert isinstance(entries, list)
        forked = [e for e in entries if e[4] > 0]
        assert forked, "no forked-stream operations in the EP manifest"
