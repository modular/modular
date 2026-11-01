# Running an EFA disaggregated-inference benchmark

This is the manual walkthrough for running a DeepSeek V3.1 NVFP4 1P1D
disaggregated-inference (DI) benchmark across two AWS B200 Coder
workspaces over EFA. Use it whenever you need to bring up DI on this
hardware combo from scratch — for validation, profiling, or
investigating an EFA NIXL regression.

The companion script
[`setup_di_workspaces.py`](setup_di_workspaces.py) handles the one-time
inter-node SSH, `/etc/hosts`, and repo-clone plumbing. Run it once per
fresh pair of workspaces before following this guide.

## Automated lane

[`run_dist_from_local.py`](../../../../max/examples/internal/di/run_dist_from_local.py)
automates steps 1-6 below into one command: given a git ref, it pins both
workspaces to the same commit, builds, launches, polls for ready, runs the
always-on sanity probe, checks for the errors in
[§5](#5--confirm-there-were-no-nixlefa-errors), and tears down — printing a PASS
/ WARN / FAIL / BUILD-FAIL / INFRA-BLOCKED verdict. Pass
`--bench-profile benchmark --bench-workload-config <yaml>` to also run a real
load test — e.g. `utils/benchmarking/serving/workloads/synthetic-75k300.yaml`,
the same workload YAML the Mammoth CI quick-test uses. Its own
`max-concurrency`/`num-prompts`/`max-benchmark-duration-s` are the defaults,
each overridable with the matching `--bench-*` flag; there's no equivalent for
the inline-flag 300-prompt benchmark in [§4](#4--run-the-300-prompt-benchmark) —
that one stays manual.

```bash
# cross-node (two Coder workspaces, EFA)
python3 max/examples/internal/di/run_dist_from_local.py <prefill-ws> <decode-ws> \
  --ref <branch-or-pr/N> \
  --config max/examples/internal/di/configs/kimi-k2_7-ep-tp-dp-fp4-eagle3-d3-internode-efa.yaml

# intranode (one workspace, no EFA needed)
python3 max/examples/internal/di/run_dist_from_local.py <workspace> \
  --config max/examples/internal/di/configs/eagle-llama-1p1d-intranode.yaml
```

It carries its own copy of the SSH / `/etc/hosts` / key-exchange
plumbing (in `coder_workspace.py`, next to the driver), so a fresh
cross-node pair bootstraps without running `setup_di_workspaces.py`
separately first.

Two flags matter for repeated runs: `--teardown none` plus `--attach
<run-id>` sends more traffic at a server that is already up, skipping
bring-up entirely, and `--lease-deadline` warns before the GPU lease
runs out. Both are documented in
[the DI README](../../../../max/examples/internal/di/README.md).

Reach for the manual steps below instead when you
need to attach a debugger or profiler to a live transfer, iterate
interactively on a fix, or the automated lane's own verdict needs
triage — its failure messages point back at the
[Common errors](#common-errors) table here.

## Prerequisites

- Two `aws-b200-workspace-pod` Coder workspaces — one prefill, one decode.
  They must be reachable via `coder ssh` from your laptop.

- Run the setup script once per fresh pair of workspaces (from your
  laptop):

  ```bash
  python3 MLRT/docs/Driver/NIXL/setup_di_workspaces.py <prefill-ws> <decode-ws>
  ```

  The script handles all the cross-node plumbing:
  - Starts each workspace if it's stopped (with confirmation).
  - Clones the Modular monorepo to `~/code/modular` on each workspace
    if it's not already there (tries `git@github.com:modularml/modular`
    first, falls back to HTTPS); also runs `git submodule update --init
    --recursive`.
  - Generates and cross-installs ed25519 SSH keys for non-interactive
    SSH between the pods.
  - Adds `/etc/hosts` entries on each side: the peer's pod hostname
    plus a friendly `b200-prefill` / `b200-decode` alias.
  - Adds `StrictHostKeyChecking accept-new` ssh_config blocks for those
    aliases so mpirun's batch-mode SSH succeeds the first time.
  - Verifies bidirectional non-interactive SSH and prints the
    `DECODE_HOST` / `PREFILL_HOST` env vars you need below.

- A recent enough `main` (or feature branch) checked out on both
  workspaces. The setup script leaves a fresh clone on `main`; if you
  need a different branch, after the script finishes:

  ```bash
  cd ~/code/modular
  git fetch origin
  git checkout <branch>
  git submodule update --init --recursive
  ```

  **Working from a local worktree (WIP / unpushed changes)?** Skip the
  git checkout above. Instead, use the rsync command printed by
  `setup_di_workspaces.py` at the end of setup to continuously sync
  your local worktree to the prefill node:

  ```bash
  # Run on your local machine, from the root of your worktree
  ./utils/rsync_coder_hosts.sh --smart --delete main.<prefill-ws>.<owner>.coder
  ```

  `--smart` re-syncs automatically on every file change (inotify on
  Linux, fswatch on macOS); `--delete` keeps the remote an exact mirror.
  The exact command with your workspace name filled in is printed by
  `setup_di_workspaces.py` at the end of its run.

## 1 · Sanity-check both workspaces have free GPU memory

On both workspaces:

```bash
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader
```

Output should be empty. If it lists PIDs, kill them with
`sudo kill -KILL <pid>`. Defunct/zombie Python processes from a prior
crashed run will still hold per-process GPU memory until reaped, and a
plain `kill` won't budge them — use root.

## 2 · Start the launcher on the prefill workspace

`coder ssh <prefill-ws>` in, then:

```bash
export DECODE_HOST=<decode-pod-hostname>      # printed by setup_di_workspaces.py
export PREFILL_HOST=<prefill-pod-hostname>    # printed by setup_di_workspaces.py

cd ~/code/modular

NIXL_LOG_LEVEL=DEBUG \
MODULAR_PATH="$PWD" \
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=95 \
MODULAR_NIXL_TRANSFER_BACKEND=libfabric \
FI_EFA_USE_DEVICE_RDMA=1 \
./bazelw run //max/examples/internal/di:run_dist -- \
  --config-yaml="$PWD/max/examples/internal/di/configs/deepseekv31-nvfp4-1p1d-internode-efa.yaml" \
  --start-proxy-server --proxy-port 9999 \
  --hosts "$DECODE_HOST,$PREFILL_HOST"
```

Keep this terminal in the foreground. Wait until the launcher's
combined output stream shows both:

- `🚀 Headless server ready` — prefill side ready (locally on this node)
- `🚀 Server ready on http://:::8000` — decode side ready (lines prefixed
  with `[b200-decode:decode_worker-0]`)

First-time model load + graph capture is ~5–8 minutes.

## 3 · Quick sanity probe

In a second `coder ssh` session into the prefill workspace:

```bash
curl -sS -m 30 http://localhost:9999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/DeepSeek-V3.1-NVFP4","messages":[{"role":"user","content":"Hi"}],"max_tokens":4}'
```

A healthy system returns a JSON chat completion in 100–500 ms.

## 4 · Run the 300-prompt benchmark

The command below stresses the EFA KV-transfer path with realistic DI
load: 300 long synthetic prompts (60–75k tokens each), 8 in flight,
streaming chat-completion responses. It's a useful baseline both for
validating that DI works and for spotting regressions in NIXL/libfabric
behavior.

In the same second session:

```bash
cd ~/code/modular

./bazelw run --config=disable-mypy //max/python/max/_entrypoints:pipelines -- benchmark \
  --backend modular \
  --model nvidia/DeepSeek-V3.1-NVFP4 \
  --endpoint /v1/chat/completions \
  --host localhost \
  --port 9999 \
  --collect-gpu-stats \
  --collect-server-stats \
  --dataset-name synthetic \
  --random-input-len "U(60000, 75000)" \
  --random-output-len "U(240, 300)" \
  --random-sys-prompt-ratio 0.96 \
  --request-rate inf \
  --num-prompts 300 \
  --max-concurrency 8 \
  --skip-test-prompt
```

Expected runtime is ~3 minutes once the pipeline is warm.

A healthy run prints, near the end:

- `Successful requests: 284` — the bench excludes
  `skip_first_n_requests=8` and `skip_last_n_requests=8` from the timing
  percentiles (16 total), but all 300 actually complete.
- `Failed requests: 0`
- `finished benchmark run: Success.`

## 5 · Confirm there were no NIXL/EFA errors

After the bench finishes (or if it fails), scan both worker logs:

```bash
echo "EINTR:         $(cat /home/ubuntu/code/modular/decode_worker.log /home/ubuntu/code/modular/prefill_worker.log | grep -cE 'fi_cq_read returned error|Interrupted system call')"
echo "INVALID_PARAM: $(cat /home/ubuntu/code/modular/decode_worker.log /home/ubuntu/code/modular/prefill_worker.log | grep -cE 'NIXL_ERR_INVALID_PARAM|InvalidParamError|No rails have both')"
echo "prov_errno=7:  $(cat /home/ubuntu/code/modular/decode_worker.log /home/ubuntu/code/modular/prefill_worker.log | grep -cE 'prov_errno=7|EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS')"
```

`cat | grep -c` rather than `grep -c file1 file2` because the latter
prints `path:count` lines that confuse arithmetic.

On a healthy DI stack, all three counters are `0`. Any of them being
non-zero points at a NIXL/libfabric problem — see
[Common errors](#common-errors) for the most-seen symptoms and where
they come from.

## 6 · Tear down

`Ctrl-C` the launcher in the first SSH session; that cascades a
`SIGTERM` to mpirun, both workers, and the proxy. Then from your
laptop:

```bash
coder stop -y <prefill-ws>
coder stop -y <decode-ws>
```

The Coder workspaces auto-stop after about an hour of idle anyway, but
explicitly stopping them frees shared resources sooner.

## Common errors

| Symptom                                                                                                       | Likely cause                                                                                                                                                                                                                                | Fix                                                                                                                                                                                                                                                                   |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Model size exceeds available memory ...`                                                                     | Zombie GPU consumers from a prior crashed run still pinning per-process GPU memory                                                                                                                                                          | Run the `nvidia-smi --query-compute-apps` check from §1; `sudo kill -KILL <pid>` for any PID listed. Defunct/zombie processes need root to be reaped.                                                                                                                 |
| `Connection refused` from the bench client                                                                    | Launcher / workers died before the bench connected                                                                                                                                                                                          | Check the launcher terminal for the upstream traceback; the bench's error itself is just the symptom.                                                                                                                                                                 |
| `Host key verification failed` from mpirun                                                                    | Pod IDs changed since `~/.ssh/known_hosts` was last populated                                                                                                                                                                               | Re-run `setup_di_workspaces.py` — it adds `StrictHostKeyChecking accept-new` blocks for the aliases.                                                                                                                                                                  |
| `ssh: Could not resolve hostname workspace-...`                                                               | `/etc/hosts` entry stale (peer pod restarted with a new ID)                                                                                                                                                                                 | Re-run `setup_di_workspaces.py`; it detects the mismatch and offers to replace the entry.                                                                                                                                                                             |
| `fi_cq_read returned error -4 ... Interrupted system call` (looping, with all CPUs pegged)                    | The libfabric CQ poll is treating `-FI_EINTR` (a signal-interrupted syscall, common with OpenMPI on EFA) as a fatal CQ error instead of retrying. Should be handled in `progressCompletionQueue`.                                           | Make sure the vendored upstream NIXL pin includes the `-FI_EINTR` retry handling in the libfabric plugin's `libfabric_rail.cpp`.                                                                                                                                      |
| `No rails have both local mr and remote key registered` followed by `NIXL_ERR_INVALID_PARAM` crashing prefill | Cross-GPU DI transfer where prefill GPU N picks rail N to send but decode GPU M ≠ N only registered rail M, leaving zero overlap. VRAM should register on every rail (like DRAM does) so any sender's local rail finds a valid rkey.        | Make sure your branch's `selectRailsForMemory(VRAM_SEG)` in `libfabric_rail_manager.cpp` returns the full rail set rather than only the GPU-local rail.                                                                                                               |
| `prov_errno=7` (`EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS`) events accumulating                            | The receiver's NIC rejected an RMA write because the sender's address wasn't yet `fi_av_insert`'d in the receiver's AV when the first `fi_writedata` arrived. This is the classic AV-insert race on FI_OPT_EFA_HOMOGENEOUS_PEERS endpoints. | A small handful per run is usually tolerable (the cascade-fix in the libfabric backend keeps a single bad write from poisoning the rail). Persistent floods point to a missing or broken MR-ready control-channel handshake before the first writedata to a new peer. |

## Kimi-K2.7-Code-NVFP4 multi-turn concurrency sweep

### Starting the server

On the prefill workspace:

```bash
export DECODE_HOST=<decode-pod-hostname>
export PREFILL_HOST=<prefill-pod-hostname>

cd ~/code/modular
NIXL_LOG_LEVEL=DEBUG \
MODULAR_PATH="$PWD" \
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=95 \
MODULAR_NIXL_TRANSFER_BACKEND=libfabric \
FI_EFA_USE_DEVICE_RDMA=1 \
./bazelw run //max/examples/internal/di:run_dist -- \
  --config-yaml="$PWD/max/examples/internal/di/configs/kimi-k2_7-ep-tp-dp-fp4-eagle3-d3-internode-efa.yaml" \
  --start-proxy-server --proxy-port 9999 \
  --hosts "$DECODE_HOST,$PREFILL_HOST"
```

Wait for `🚀 Headless server ready` (prefill) and `🚀 Server ready on
http://:::8000` from the decode side before starting the benchmark.

### Benchmark command

In a second session on the prefill workspace:

```bash
cd ~/code/modular
./bazelw run --config=disable-mypy \
  //max/python/max/_entrypoints:pipelines -- benchmark \
  --backend modular \
  --model nvidia/Kimi-K2.7-Code-NVFP4 \
  --endpoint /v1/chat/completions \
  --host localhost \
  --port 9999 \
  --collect-gpu-stats \
  --collect-server-stats \
  --dataset-name synthetic \
  --random-input-len "G(2.099,22184.313);G(2.099,2126.134)" \
  --random-output-len "LN(5.852,0.961)" \
  --random-sys-prompt-ratio 0.95 \
  --warm-shared-prefix \
  --num-chat-sessions 256 \
  --random-num-turns "NB(0.525,0.1)" \
  --delay-between-chat-turns "LN(5.168,1.444)" \
  --request-rate inf \
  --model-max-length 262144 \
  --trust-remote-code \
  --seed 260 \
  --num-prompts 700 \
  --max-concurrency 4,8,16,32 \
  --skip-test-prompt
```

`--seed 260` pins the random draws so every run across all concurrency levels
sees the same prompt/turn sequence, making runs directly comparable and
results reproducible. Omit it (or change the value) to draw a fresh sequence.

Known reproduction seeds:

| Seed | Stall at | Failures |
|------|----------|----------|
| 260  | mc=32    | 6        |
| 1041 | mc=16    | 3        |

## What the launcher does, in one paragraph

`run_dist` reads the YAML config and spawns two `mpirun` jobs — one
per worker role — plus a proxy server. Each `mpirun -np 1 -H <host>:1`
SSHes (when remote) into the named host and runs
`./bazelw run //max -- serve …` there with the right
`--pipeline-role` flag and the per-rank `MAX_SERVE_*` env vars. The
proxy listens on `0.0.0.0:9999`, picks a (prefill, decode) pair per
incoming request, and forwards the chat-completion body to decode after
injecting `target_endpoint` so decode's scheduler knows which prefill
ZMQ dispatcher to talk to. All KV-cache RDMA traffic between the two
workers goes over libfabric/EFA — that's the path this benchmark
exercises.
