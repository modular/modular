# Benchmarks

This directory contains benchmarks for the Max kernels. The benchmarks are
designed to evaluate the performance of the Max kernels under various
conditions and configurations.

## Quick Performance Testing on a PR

The `.github/workflows/kbenchSmokeTest.yaml` workflow supports both single-GPU
and multi-GPU benchmarks via the `mode` input. Both modes support A/B comparison
against `main` via `compare_to_main`. Results appear in the GitHub Actions run
summary.

### Single-GPU smoke test

Runs a fixed set of benchmarks (gemm shapes, topk) on H100, B200, and MI355.
The benchmark list is hardcoded in the workflow matrix.

```bash
# Run on your branch
gh workflow run kbenchSmokeTest.yaml --ref my-branch -f mode=single-gpu

# A/B compare against main
gh workflow run kbenchSmokeTest.yaml --ref my-branch -f mode=single-gpu -f compare_to_main=true
```

### Multi-GPU smoke test

Runs collective/comm benchmarks on 8xH100, 8xB200, and 8xMI355. The benchmark
list is configurable via `benchmark_yamls` (defaults to allreduce +
reduce-scatter).

```bash
# Run defaults on your branch
gh workflow run kbenchSmokeTest.yaml --ref my-branch -f mode=multi-gpu

# A/B compare against main
gh workflow run kbenchSmokeTest.yaml --ref my-branch -f mode=multi-gpu -f compare_to_main=true

# Run specific benchmarks only
gh workflow run kbenchSmokeTest.yaml --ref my-branch \
  -f mode=multi-gpu \
  -f compare_to_main=true \
  -f benchmark_yamls="max/kernels/benchmarks/gpu/bench_allreduce_smoke.yaml"
```

### Monitoring a run

```bash
# Watch live
gh run watch <run-id>

# View summary after completion
gh run view <run-id>
```

### Claude skill

`/bench-kernel-pr` auto-detects which mode to trigger based on your
branch's changed files and handles the `gh workflow run` invocation + PR
comment. Run it in Claude Code with no arguments for auto-detection, or pass
`single-gpu` / `multi-gpu` to override.

For details on `kbench` itself, see [autotune/README.md](autotune/README.md)
and [Benchmarking Mojo kernels with kbench](/max/docs/kernel-benchmarking.md).
