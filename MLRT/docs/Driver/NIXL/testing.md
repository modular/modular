# NIXL testing

NIXL allows us to share memory between GPUs on different hosts for
KV-cache transfer in disaggregated inference. Its tests span from a
pure library check up to end-to-end two-node serving. They
are organized by one rule: a lane **gates** (its failure means the code
is broken) only when it runs where hardware is guaranteed — remote
execution, a dedicated GPU runner, or a runner-label queue that waits
for a machine. Lanes that share a contended machine pool, where a run
can fail simply because no node was free, are **informational**: their
results are reported, not gated. Splitting off the informational lanes
is only necessary because of GitHub Actions restrictions. A CI system
that allows a gating run wait on those pod-pool jobs would let them gate
too.

## Lanes

| Lane                                | What it checks                                                                                                                     | Where it runs                                         | Cadence                | Gates?             |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|------------------------|--------------------|
| Basic NIXL behaviors                | real NIXL send/recv/read over host DRAM (CPU UCX plugin)                                                                           | remote execution, no GPU                              | pre-submit             | Yes                |
| GPU unit, bindings, transfer engine | MLRT Core/UCX unit tests; `max._core.nixl` agent/memory/transfer; `KVTransferEngine` single-GPU send/recv                          | `validateNIXLSync`                                    | nightly                | Yes                |
| Multi-GPU transfer engine           | multi-GPU send/recv, DP/TP, heterogeneous MLA, plugin preload                                                                      | `nightlyMultiGPUTest` on B200 and MI355               | nightly                | Yes                |
| Single-node perf smoke              | NVIDIA UCX + EFA loopback, NVIDIA UCX rc (IB loopback) and AMD UCCL bandwidth floors — catches transport-selection regressions     | `modrunner-b200-efa-2x` + `-b200-ib-2x` + `-mi355-2x` | nightly                | Yes                |
| Shared-pool                         | 4-row cross-node EFA KV smoke + the nightly's `nixl-validation-1p1d-*` Mammoth benchmarks                                          | shared B200 and MI355 pod pools                       | dispatched by the gate | No (informational) |
| Shared-pool (AMD)                   | 4-row cross-node RoCE KV smoke on AMD, UCCL backend                                                                                | shared AMD (MI355) pod pool                           | dispatched by the gate | No (informational) |
| Client-driven                       | Same DI bring-up + benchmark as the shared-pool lane, via `run_dist_from_local.py`; substitute when the shared pool is unavailable | personal Coder workspace(s), cross-node or intranode  | on demand              | No (informational) |

The perf smoke carries one NVIDIA row per B200 fabric CI has, and both
are load-bearing: libfabric (EFA) and cuda-verbs (InfiniBand) are
different plugins, so neither row substitutes for the other. The AWS EFA
nodes left the fleet in early September 2026 and came back mid-month,
with the Hydra InfiniBand nodes arriving in the same shuffle; the EFA row
was off in between (GEX-4168), which left libfabric with no gating
coverage. If a pool leaves again, drop its row rather than letting it
queue — a job-level `if:` cannot read the `matrix` context. Each row's
label must name its fabric (`-efa`, `-ib`): the bare `modrunner-b200-2x`
is "any B200 pair" and draws EFA hosts, where the IB row's
`UCX_TLS=rc` dies opening `ud_verbs` on an EFA device instead of
falling back.

The InfiniBand rows run under `numactl` on the GPU's own NUMA node. The
runner container spans both sockets while the pod's GPUs and rails sit on
one of them, and rc stages through host memory, so an unbound row measures
anywhere from 48 down to 19 GiB/s depending on where the kernel puts the
staging buffer. That spread, not a transport regression, is what used to
redden the 256 KiB row. If a bandwidth row ever looks bimodal again, the
row logs carry a placement dump and UCX's protocol-selection table, which
name the CPUs, the NUMA node and the rail it actually used.

## The aggregate gate

The `validateNIXLSync` is the one CI run that validates a NIXL
change end to end. Its red/green is the correctness gate — the
in-tree unit tests (`//MLRT/unittests/Driver/NIXL/...`), the binding and
transfer-engine tests, and the single-node perf smoke, all lanes that
wait for hardware to be available in CI. It runs nightly on `main` (a
standing baseline) and on demand:

```bash
gh workflow run validateNIXLSync.yaml --ref <branch-or-main>
```

It also fire-and-forgets the shared-pool lane as an informational
companion. Pass `-f run_shared_pool_tests=false` to gate on correctness
alone.

## Why the shared-pool lane is informational

The cross-node EFA smoke and the Mammoth benchmarks schedule pods onto a
small shared machine pool. When the pool is full, pods sit `Pending` past
the deployment deadline and the job fails — indistinguishable, at a
glance, from a code regression. The drivers classify that signature as
**INFRA-BLOCKED** (a warning, not a red), so a genuine failure stands
out. And because a loopback smoke does not cross the physical fabric, the
two-node EFA smoke remains the guard for fabric-level regressions: weigh
a red there against the captured pod diagnostics before treating it as a
code bug.

The AMD shared-pool lane is the same 4-row cross-node smoke on AMD GPUs
over the RoCE (Pensando) fabric, run from the ROCm serving image — the
same `nixlKvSmokeTest.yaml` workflow as the EFA lane, dispatched with
`gpu_vendor: amd` (which selects the `-amd` image, the `k8s/rocm`
manifests, and the AMD pod pool). It is the AMD counterpart of the two-node EFA
smoke and the only lane that crosses the AMD fabric end to end (the
multi-GPU transfer-engine lane runs on MI355 but same-node). Its pods
request `amd.com/gpu` plus `amd.com/nic` — the RoCE IB
fabric, whose CI availability is the analog of the Coder "Enable RoCE
networking" workspace flag. It uses UCCL, the default AMD backend, which
saturates the RoCE fabric, so it carries the same bandwidth floors as the
EFA smoke.

## Benchmarks

The **drag race** (`//utils/benchmarking/nixl:drag_race`) is an
on-demand bandwidth benchmark for a chosen backend (`ucx`, `efa`, or
`uccl`) and topology (intranode loopback or two nodes). It is not run in
CI, but it builds in presubmit and has a host-DRAM micro-run that keeps
it green. The `uccl` backend stages the AMD UCCL plugin flavor into the
harness runfiles, so it can drive UCCL over an AMD RoCE fabric without an
installed serving image.

## Related docs

- [`validate_upstream_sync.md`](validate_upstream_sync.md) — what to run
  before landing an upstream NIXL pin bump.
- [`run_efa_di_benchmark.md`](run_efa_di_benchmark.md) — manual two-node
  EFA disaggregated-inference benchmark, plus the automated
  `run_dist_from_local.py` lane.
- [`update_instructions.md`](update_instructions.md) — how to bump the
  vendored NIXL pin.
