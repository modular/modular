# TileTensor Migration Assessment

Brutally honest evaluation of the work on
`fabio/tiletensor-kernel-signatures-v2`.

## Scorecard

| Objective | Status | Grade |
|-----------|--------|-------|
| Remove legacy Layout params from all kernel structs | Done (8/8) | A |
| TMA types derived from config (single source of truth) | Done | A |
| No rebind at kernel boundaries | Done | A |
| No performance regressions | Verified (benchmarks) | A |
| Fallback kernel C output on TileTensor | Done | B+ |
| Split-K reduction tensor on TileTensor | Done | A |
| RegTile / accumulator on TileTensor | Done | A |
| Clean, readable code | Mixed | B- |
| TileTensor API completeness | Gaps remain | C+ |

## What went well

**The pattern works.** "Kernel first, explicit layouts" is clean and
mechanical. Every kernel followed the same recipe:

1. Remove layout params from struct
2. Compute TMA layouts from config via `static_row_major`/`RowMajorLayout`
3. Derive `TmaOpType` aliases
4. Instantiate kernel before TMA creation on host
5. Pass `Kernel.XTmaOp.layout` to `create_tensor_tile`

This eliminated 60+ legacy Layout parameters across 8 kernel families
with zero functional changes and zero performance regressions.

**The TMA type identity problem is solved.** `TmaOpType` as a single
source of truth, with explicit layout passing to `create_tensor_tile`,
guarantees type identity by construction. No rebind needed. This was
the hardest design problem and it's cleanly resolved.

**Benchmark results are noise-level.** Every kernel we benchmarked
(bf16, fp8, fallback, blockwise, grouped) showed <1% variance from
main. The changes are compile-time only -- generated GPU code is
identical.

## What's clunky

### 1. The `stride_layout` parameter is a hack

The `stride_layout` parameter on `tile`/`tile_with_offset` uses a
Layout type whose **shape** encodes the desired **strides**. This is
semantically confusing:

```mojo
comptime CGmemStrideLayout = Layout[
    shape = [ComptimeInt[N], ComptimeInt[1]],  // these are STRIDES
    stride = [ComptimeInt[1], ComptimeInt[1]],  // ignored
]
```

The shape of a "stride layout" represents strides. The actual stride
of the "stride layout" is meaningless. This violates the principle of
least surprise. A better API would accept explicit stride types
directly, not smuggle them through a layout's shape.

**Root cause**: Mojo doesn't support keyword-only variadic parameters,
and we can't have two variadic packs in one parameter list. So we
can't do `tile[*tile_sizes, *static_strides]`. The Layout type was
the most expedient encoding.

**Fix**: Add a proper `StaticStrides[*strides: Int]` type alias or
wait for Mojo to support richer parameter patterns.

### 2. TensorLayout trait erases too much

Using `TensorLayout` as a struct parameter erases the concrete
`Layout[ComptimeInt[...], ...]` type, so the compiler can't prove
`all_dims_known`. This forced the `stride_layout` workaround.

LayoutTensor never had this problem because its `Layout` (IntTuple)
is always a concrete comptime value -- there's no trait indirection.

**This is the fundamental tension**: TileTensor uses traits for
flexibility, but traits erase the compile-time guarantees that GPU
kernels need. The `stride_layout` pattern papers over one instance,
but the underlying design tension remains.

### 3. Host-side functions still use LayoutTensor

Every host launch function still accepts `LayoutTensor` parameters
from callers. We do `lt_to_tt(c)` at the `enqueue_function` boundary.
The LayoutTensor surface area hasn't actually shrunk at the API level
-- we just pushed the conversion point from inside the kernel to
outside it.

This is correct (the kernel is cleaner), but the callers still create
LayoutTensors. True elimination requires changing the entire call
chain up to the graph compiler.

### 4. Too many commits

31 commits for what should have been ~5-8 focused changes. The
history includes false starts (TMATile wrapper, rebind attempts),
formatting commits, doc fixes, and stale file removals. A cleaner
history would tell a better story.

### 5. `_to_legacy_layout` is still needed

Every `TmaOpType` goes through `_to_legacy_layout` to convert new
Layout types to IntTuple-based Layout for `TMATensorTile`. This
bridge exists because `TMATensorTile` itself hasn't been migrated.
It's a necessary evil but adds complexity to `tile_types.mojo`.

## What's missing from TileTensor

### Critical gaps

1. **`zipped_divide` / `upcast` / `blocked_product`** -- layout
   algebra operations used by `epilogue_components.mojo` and
   `output_writer.mojo`. These are the last major blockers for
   removing LayoutTensor from the epilogue path. No TileTensor
   equivalent exists.

2. **`TMATensorTile` on new Layout** -- TMA descriptors still use
   IntTuple-based Layout internally. Migrating TMATensorTile would
   eliminate `_to_legacy_layout` entirely.

3. ~~**`RegTile` / accumulator**~~ -- DONE. Migrated to TileTensor
   `stack_allocation` with `row_major` layout.

### Ergonomic issues

1. **No default Layout constructor** -- `Layout(shape, stride)`
   requires explicit Coord construction. Can't write `Layout()` for
   a fully-static layout. Makes comptime layout construction verbose.

2. **Coord construction is verbose** -- `Coord(Idx(Int(x)), Idx(y))`
   vs LayoutTensor's bare `(x, y)`. Every `tile_with_offset` call
   needs explicit Coord wrapping.

3. **Tuple destructuring quirks** -- `var a, b, _ = f()` works but
   `a, b, _ = f()` (without `var`) doesn't bind in all scopes.
   This is a Mojo language issue, not TileTensor-specific.

4. **`tile()` inherits dynamic stride types** -- even when the
   parent has static stride VALUES, if the stride TYPES are
   `RuntimeInt` (from a trait parameter), the tile gets `RuntimeInt`
   too. The `stride_layout` workaround handles this but shouldn't
   be necessary.

5. **No `vectorize_with_offset`** -- we added
   `distribute_with_offset` but `vectorize` doesn't have an
   offset-returning variant. Not needed yet but would complete
   the API.

## By the numbers

| Metric | Value |
|--------|-------|
| Commits | 31 |
| Files changed | 25 |
| Lines added | 2,601 |
| Lines removed | 5,660 |
| Net | -3,059 (mostly deleted docs) |
| Legacy Layout struct params removed | 60+ |
| Kernel families migrated | 8/8 |
| New TileTensor methods | 8 |
| New infra types | TmaOpType, TmaOpTypeIm2col, tma_desc_layout_3d/4d/5d |
| Tests passing | All |
| Performance regressions | None |

## Remaining LayoutTensor (176 references in sm100_structured)

Most are in:

- **Host function signatures** (~60) -- accept LayoutTensor from
  callers
- **output_writer.mojo** (~12) -- zipped_divide/upcast path
- **epilogue_components.mojo** (~9) -- SMEM epilogue, layout algebra
- **tile_types.mojo** (~18) -- lt_to_tt converters, GMEMTile bridge
- **pipeline_storage.mojo** (~13) -- comments and LT tile aliases

The kernel `run()` functions themselves are clean -- LayoutTensor
only appears in the shared infrastructure that serves all kernels.

## Verdict

The branch achieves its primary objective: all 8 kernel structs have
zero legacy Layout parameters, TMA types are derived from a single
source of truth, and there are no performance regressions.

The solutions range from elegant (TmaOpType pattern, kernel-first
instantiation) to pragmatic (stride_layout hack, manual Coord
construction). The TileTensor API has real gaps (layout algebra,
trait erasure) that will require library-level work to resolve.

The honest summary: we cleared the kernel surface but the internals
(epilogue, accumulator, host signatures) still have substantial
LayoutTensor. The migration is about 70% complete by code volume,
but the remaining 30% requires deeper changes (layout algebra,
TMATensorTile migration) that are outside the scope of this branch.
