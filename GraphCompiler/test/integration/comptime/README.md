# Compile-time sniff tests

Hand-written `mo.graph` fixtures in this directory for manually sniffing
compile time, kernel dedup, and generated-Mojo size on a machine you
control -- that's hard to do reliably in CI (noisy shared runners, no stable
baseline machine). Each file's own `RUN:` line only parses (`mo-opt %s |
FileCheck %s`, no GPU needed), which is what keeps them from going stale;
the actual sniffing is a manual recipe against the full `--gc-pipeline`
compile, below (needs a GPU, since these fixtures are GPU-typed).

## Manual sniff recipe

Build `mo-opt` once, then iterate directly against it (`GraphCompiler/docs/
InvestigatingCompileTimeRegressions.md` covers the general version of this
workflow against real models):

```bash
./bazelw build //GraphCompiler/tools/mo-opt
MO_OPT=bazel-bin/GraphCompiler/tools/mo-opt/mo-opt
FIXTURE=GraphCompiler/test/integration/comptime/large-transformer-stack-static.mlir
```

**Total compile time** -- MLIR's pass-timing infra, already wired into
`mo-opt`:

```bash
$MO_OPT $FIXTURE --gc-pipeline --mlir-timing -o /dev/null
# ... per-pass breakdown ...
# Total Execution Time: N.NNNN seconds
```

**Total kernels (before dedup)** -- kernel-stub dedup
(`MergeIdenticalFuncs`) runs inside `EmitMojo`'s own nested pass
pipeline, not as a top-level pass, so it isn't directly targetable with
`-mlir-print-ir-before/after`. Count `mogg.func` ops right before
`EmitMojo` runs instead:

```bash
$MO_OPT $FIXTURE --gc-pipeline -o /dev/null \
  -mlir-print-ir-before=mogg-emit-mojo 2>&1 >/dev/null \
  | grep -c 'mogg.func'
```

**Unique kernels (after dedup) and lines of generated Mojo** -- by the time
`EmitMojo` finishes it has already erased every `mogg.func` (their
bodies became emitted Mojo text), so "after `mogg-emit-mojo`" reads 0, not
the deduped count. Dump the Mojo and count distinct kernel function
definitions instead -- one `def stub_N` per surviving unique kernel:

```bash
$MO_OPT $FIXTURE --gc-pipeline -o /dev/null \
  -mlir-print-ir-after=mogg-emit-mojo &> /tmp/out.mgp.mlir
$MO_OPT /tmp/out.mgp.mlir -mogg-dump-mojo -o /dev/null &> /tmp/out.mojo
grep -c '^def stub_' /tmp/out.mojo   # unique kernels
wc -l /tmp/out.mojo                  # generated Mojo LOC
```

## Pitfalls

- These recipes never touch the MEF/MOGG caches (`mo-opt` compiles a bare IR
  file directly, no `InferenceSession`/model-cache layer involved), so
  there's no need to clear caches between runs the way you would when
  investigating a real pipeline's compile time.
- `-mlir-print-ir-before=<pass>`/`-mlir-print-ir-after=<pass>` only see
  passes that run on the *top-level* pass manager built by `--gc-pipeline`.
  Passes run inside another pass's own nested `PassManager` (like
  `MergeIdenticalFuncs` inside `EmitMojo`) are invisible to these
  flags -- bracket the *outer* pass instead, as above.
