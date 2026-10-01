# TMEM tensor engine

Status: engine, host tests, B200 round-trip tests, and batched copies in both
directions are done (steps 1 to 4 below); bridges and fusion are open.

This document describes a `TensorEngine` that lets a `TileTensor` view
Blackwell Tensor Memory (TMEM). It records the hardware facts the design rests
on, the decisions that are not obvious from the trait, an API sketch, the
implementation order, and the open questions.

## Motivation

On SM100 the MMA accumulator lives in TMEM, not in registers. Every kernel
that consumes an accumulator therefore hand-codes the same three things:
the 16-bit row-packed address format, the split into upper and lower
fragments, and the lane placement implied by the MMA shape. The kernel tree
holds three parallel wrappers for this today:

- `TmemTensor`, `TmemAddress`, and `TmemFragments` in
  `max/kernels/src/linalg/matmul/gpu/sm100_structured/structured_kernels/tmem.mojo`,
  parameterized on the legacy `Layout` type.
- `TMemTile` in `attention_utils.mojo` under
  `max/kernels/src/nn/attention/gpu/nvidia/sm100/`, a raw `UInt32` address
  with an ad-hoc `_tmem_offset()` helper.
- Open-coded `TmemAddress` arithmetic in the `mha_depth512` softmax and
  correction warps.

`TileTensor` already abstracts its storage behind the `TensorEngine` trait in
`max/kernels/src/layout/tensor_engine.mojo`. A TMEM engine makes an
accumulator an ordinary tile: it gets `tile()`, `copy_from()`, hierarchical
`TensorLayout` types, and the fusion graph API for free, and the three
wrappers above collapse onto one typed view.

## Hardware facts the design depends on

These come from the PTX ISA for `tcgen05` and from the wrappers in
`max/mojo/max/gpu/compute/arch/tcgen05.mojo`.

- **Address format.** A TMEM address is a 32-bit integer whose upper half is
  the lane (row, 0 to 127) and whose lower half is the column. The existing
  lower-fragment constant `16 << 16` is "lane 16, column 0".
- **Capacity.** 512 columns of 128 lanes per CTA, each cell 32 bits.
  Allocation is per CTA via `tcgen05_alloc()`, which writes the base address
  into a shared-memory slot.
- **Warp to lane restriction.** Warp `w` of a warpgroup may access only lanes
  `32 * (w % 4)` through `32 * (w % 4) + 31`. No thread can address a lane
  outside its warp's quadrant.
- **Collective access.** `tcgen05.ld` and `tcgen05.st` are `.sync.aligned`
  warp instructions. The address operand carries the warp's base lane; the
  hardware assigns lanes to threads according to the shape:
  - `32x32b`: thread `t` receives lane `base + t`, `repeat` consecutive
    32-bit columns.
  - `16x256b`: 16 lanes, two threads per lane, each thread holding
    interleaved 4-column chunks. This is the register layout the current
    epilogues feed to `st_matrix`; `store_fragment_to_smem()` in
    `epilogue_components.mojo` encodes the thread mapping.
- **Convergence.** The `.aligned` qualifier requires every thread of the warp
  to execute the same `tcgen05.ld` or `st` together. A divergent branch
  between two of the engine's loads is undefined behavior and hangs in
  practice, so kernels keep the code around engine loads and stores
  warp-uniform, and lanes that hold no logical row still issue them.
- **Asynchrony.** Both instructions complete asynchronously. Loaded registers
  are valid only after `tcgen05_load_wait()`; stored registers may not be
  reused, and the TMEM contents may not be read, before
  `tcgen05_store_wait()`.
- **Widths.** `repeat` and the resulting per-thread width are powers of two
  from 1 to 128, and the wrappers accept 4-byte element types only. Two-byte
  types go through `pack::16b`.
- **Placement depends on the MMA shape.** For `cta_group=1` and M=128, row
  `r` is lane `r`. For `cta_group=1` and M=64 each warp uses only the first
  16 lanes of its quadrant, so row `r` is lane `32 * (r // 16) + r % 16`.
  This is what the `is_lower_required` flag in `tmem.mojo` encodes. The
  `cta_group=2` placements must be derived from the ISA data-path figures.

## What the trait requires

A conforming engine (see `TensorEngine` in `tensor_engine.mojo`) provides:

- `StorageType[dtype, origin, address_space]`, a `TrivialRegisterPassable`
  handle. It does not have to be a pointer; `DevicePointerEngine` uses a
  `DevicePointer` and ignores `address_space`. The TMEM handle does the same:
  no `AddressSpace` value describes TMEM (PTX has no state space for it; the
  `tcgen05` address is a plain 32-bit register), so `TMemStorage` carries
  only `dtype` and `origin`.
- `unsafe_ptr(storage)`, the raw scalar pointer every element load and store
  on a `TileTensor` goes through. It cannot raise; an engine whose storage
  has no pointer rejects the call with a `comptime assert`, which makes
  element access on such a tile a compile error rather than a runtime one.
- `offset(storage, Coord)` with an `OffsetResultType`, `distance()`, and
  `unsafe_cast()`.
- `copy_from()` between two `(storage, layout)` pairs, possibly across
  engines, and `copy_to()`, its source-side counterpart.
  `TileTensor.copy_from(other)` calls `other`'s engine `copy_to`, whose
  default forwards to the destination engine's `copy_from`, so a
  pointer-less engine can intercept both directions of a copy.
- `element_size` and `_BASE_TYPE_NAME`.

`TileTensor.offset` and `tile()` compute `layout(coord)` and pass the flat
result to `Engine.offset()`. The layout therefore decides what an offset
means to the engine, and the engine decides how to turn it into an address.

`TensorOps` (elementwise arithmetic) is a separate trait. `TileTensor` only
exposes `__iadd__()` and friends when the engine conforms to it, and it
requires both operands to share an engine.

## Design

### The layout is the lane-by-column grid

TMEM is a grid of `TMEM_NUM_LANES` (128) lanes by `TMEM_NUM_COLS` (512)
32-bit columns. A layout over the engine places elements on that grid: its
flat index walks the grid lane-first, so a lane stride is 1 and a column
stride is `TMEM_NUM_LANES`. The whole accumulator is `(128, 512):(1, 128)`;
an `N`-column allocation is `(128, N):(1, 128)`, a sub-tile of it. The
hardware's address encoding, lane in the upper 16 bits and column in the
lower, never appears in a layout. The storage handle holds the encoded
`UInt32`, and:

- `offset()` decodes the handle to a grid index, adds the coordinate sum,
  and re-encodes the result.
- `distance()` subtracts two decoded grid indices.
- Lane placement lives in the layout, in grid units. Nested `TensorLayout`
  shapes express the quadrant placements without any engine-side special
  cases.

Lanes are the stride-1 dimension, as in CuTe's TMEM layouts, even though
a copy moves along a lane's columns. Every allocation has exactly 128 lanes
and `tcgen05.alloc` only chooses a column count, so a lane-first grid gives
every tile the same conversion layout, `(128, N):(1, 128)`, and that layout
survives a part with a different column budget (Rubin raises the maximum to
576). It also keeps the conversion cheap: with the lane count a power of two,
`_encode` and `_grid_index` are a shift and a mask, where a column-first grid
divides by the column count. The same scheme fits Trainium, whose engines
resolve a logical index into a partition and an offset rather than an
integer address.

An earlier draft used a row stride of `1 << 16` so that the flat offset was
the address delta itself. That leaked the encoding into every layout, made a
dense accumulator look strided, and would not carry over to other non-flat
memories such as Trainium's SBUF and PSUM, whose engines have to resolve
addresses themselves. A second draft walked the grid column-first, with a
row stride of 512; it was logical but baked the column count into every
layout. Resolving addresses inside the engine keeps the layout logical.

| MMA configuration    | Shape          | Stride           |
|----------------------|----------------|------------------|
| `cta_group=1`, M=128 | `(128, N)`     | `(1, 128)`       |
| `cta_group=1`, M=64  | `((16, 4), N)` | `((1, 32), 128)` |
| `cta_group=2`, any M | to be derived  | to be derived    |

Layouts are two-dimensional, lanes by columns; the copy path asserts a rank
of 2. Nested modes inside those two dimensions, as in the M=64 row, are how
placements are spelled. How a logically three- or four-dimensional tile
should map onto the two physical dimensions is an open question, so it is
rejected for now rather than given a default.

`_get_index_type()` picks `int32` for these layouts because the static cosize
stays below 2^31. A layout with runtime dimensions would switch to `int64`
silently, so TMEM layouts should be fully static.

### No TMEM address space

Review asked whether TMEM should get its own `AddressSpace` value, the way
Trainium's SBUF and PSUM engines do. LLVM's NVPTX backend does model one
(the `tcgen05` intrinsics take `ptr addrspace(6)`), but PTX has no such
state space and nothing is ever dereferenced there, so a stdlib constant
would only type a pointer that can never exist. The `Engine` parameter
already distinguishes a TMEM tile from register, shared, and global tiles,
so the engine leaves the trait's `address_space` slot unused, as
`DevicePointerEngine` does. A shared convention for non-flat memories is
worth deciding together with the SBUF/PSUM engine rather than here.

### One thread owns one lane

Because access is collective and the hardware assigns lanes, a copy between
a TMEM tile and a register tile means "the consecutive columns of the lane
this thread owns" and lowers to `tcgen05.ld.32x32b.x{width}` or the matching
`st`. The address passed to the instruction must be the warp's base lane,
never `base + lane_id`.

Consequently a thread does not hold the warp's `(32, N)` tile. It holds row
0 of it, `warp_tile.tile[1, N](Coord(Idx[0], Idx[0]))`, whose storage is the
unchanged warp base address. No engine-specific helper is needed: a `(1, N)`
sub-tile at row 0 is already a view with the right storage and a column
stride of `TMEM_NUM_LANES`. Regular `tile()` remains valid at warp
granularity: a `(128, N)` accumulator tiled as `tile[32, N](Coord(warp_id,
0))` advances the grid index by `32 * warp_id`, which the engine encodes as
lane `32 * warp_id`, exactly the quadrant base the hardware requires. For the
nested M=64 layout the same thing is spelled with a hierarchical coordinate,
`Coord(Coord(Idx[0], warp_id), Idx[c])`.

A copy through the `(32, N)` warp tile itself would be wrong, since the
hardware adds the lane again. The engine rejects it at compile time: the TMEM
operand of a copy must be a run of consecutive columns (`_is_lane_run()`, a
gap-free check with a base stride of `TMEM_NUM_LANES` that ignores extent-1
dimensions, so a `(1, N)` row view qualifies whatever its lane stride) and
hold at most `TMEM_NUM_COLS` elements. A view spanning more than one lane
fails the first test. The row-0 view is the
only per-thread handle kernels should hold.

### The load shape follows the element type

The engine has no parameters. The `tcgen05` access shape is derived where
an instruction is issued: its lane count is always 32, one thread per lane,
which is what makes a copy mean "this thread's row" and matches the trait
directly; its bits per lane are the element type's width, so a float32 tile
moves in `32x32b`. A TMEM allocation always has 128 lanes with 32 visible
to each warp, so there is nothing for a lane-count parameter to vary, and a
bits parameter would only restate the dtype. `16x256b` changes the thread
mapping and the register layout, and the existing epilogues depend on that
layout for `st_matrix`; it is a different engine, not a parameter of this
one.

### Copies are batched behind one wait, 64 columns at a time

The engine's only data path is the copy. `copy_to()` issues the `tcgen05.ld`
instructions for a row, calls `tcgen05_load_wait()`, then stores through the
destination engine's pointer. `copy_from()` gathers the row through the
source engine's pointer, issues the `tcgen05.st` instructions, and calls
`tcgen05_store_wait()`, because inline assembly does not model the
register-reuse hazard. The row is split into power-of-two chunks, largest
first, since those are the repeat counts the ISA accepts; a 32-column row is
one `x32` instruction.

Two limits bound the registers a copy needs. No instruction names more than
64 registers per thread: the ISA allows `x128`, but a 128-register operand
list needs 128 consecutive registers and `ptxas` rejects it with C7602 in a
kernel that cannot spare them, which is why the SM100 attention kernels cap
their `tcgen05.ld` width at 64 (`ld_splits` in `softmax_warp.mojo`). And a
waiting copy moves the row in 64-column slices, each gathered or loaded,
waited on, and scattered before the next, so a row of any width holds at
most 64 staging registers live; a row of up to 64 columns is one slice with
one wait, as before. Per-slice waits are what make the store side safe: a
slice's staging registers are dead at its wait, so the next slice's gather
can reuse them. The codegen test asserts that a 128-column row is two `x64`
instructions and two waits per direction.

`copy_from_async()` and `copy_to_async()` issue the same instructions
without the wait, and `TMemEngine.wait_store()` / `wait_load()` wait for
everything the thread has issued, so several copies can share one wait. An
async copy stages its whole row at once, since without a wait between slices
the store side could not reuse staging registers, so the engine rejects an
async row wider than 64 columns at compile time. A wider row is tiled into
slices with one async copy each; the staging registers of every slice stay
live until the shared wait, and that is the caller's trade.
`tmem_copy_async(dst, src)` is the tile-level spelling for either
direction. The caller owns the hazards the wait normally covers: a source
of an async store must not be modified before `wait_store()`, and the
destination of an async load must be a register tile that is not read
before `wait_load()`, since the loaded registers hold stale data until
then. This mirrors the `load_fragments()` / `wait_load()` pattern in
`TmemTensor`.

The shared `_copy_from()` loop in `tensor_engine.mojo` is not used: it reads
and writes through `unsafe_ptr`, which TMEM does not have.

### Conform to `TensorEngine` only

TMEM has no ALU. The right idiom is copy into a register tile, compute there,
copy back, and the same-engine gate on in-place ops already forces that. The
engine therefore does not conform to `TensorOps`, and `unsafe_ptr()` is a
`comptime assert` that rejects any instantiation, so element loads and
stores on a TMEM tile do not compile. A TMEM-to-TMEM copy is rejected the
same way; it has to stage through a register tile.

### Allocation stays outside the engine

`TensorEngine` never owns memory. `TmemAllocation` keeps the alloc, the
shared-memory address handshake, the allocation lock, and the dealloc barrier.
The engine only wraps the address the allocation hands out.

### Host compilation

Instantiating any `tcgen05` wrapper on a non-Blackwell target fails a
compile-time assertion, and the engine relies on that rather than guarding
the copies with a runtime `abort()`. Only instantiated methods are checked,
so the layout package and the host tests, which exercise `offset()`,
`distance()`, `unsafe_cast()`, and tiling but never a copy, still compile
everywhere. Unsupported operations follow the same rule: `unsafe_ptr()` is a
`comptime assert False`, so a kernel that reaches for a pointer to TMEM
fails to build instead of failing at runtime.

## API sketch

```mojo
comptime TMEM_NUM_LANES = 128
comptime TMEM_NUM_COLS = 512


struct TMemStorage[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
](TrivialRegisterPassable):
    """Encoded TMEM address: lane in the upper 16 bits, column in the lower."""

    var addr: UInt32


struct TMemEngine(TensorEngine):
    comptime element_size = 1
    comptime _BASE_TYPE_NAME: StaticString = "TMemEngine"

    comptime StorageType[
        mut: Bool, //, dtype: DType, origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = TMemStorage[dtype, origin]

    comptime OffsetResultType[
        offset_types: TypeList[Trait=CoordLike, ...]
    ]: TensorEngine = Self

    @staticmethod
    def unsafe_ptr[...](storage: Self.StorageType[...]) -> Pointer[...]:
        comptime assert False, "tensor memory has no pointer representation"

    # TMEM row <- register or shared-memory tile: gather through the source
    # pointer, one `tcgen05.st` per power-of-two chunk of at most 64
    # registers, one wait per 64-column slice.
    @staticmethod
    def copy_from[...](storage: Tuple[Self.StorageType[...], L1],
                       other: Tuple[OtherEngine.StorageType[...], L2]): ...

    # Register or shared-memory tile <- TMEM row: one `tcgen05.ld` per chunk,
    # one wait per 64-column slice, then scatter through the destination
    # pointer.
    @staticmethod
    def copy_to[...](storage: Tuple[Self.StorageType[...], L1],
                     other: Tuple[OtherEngine.StorageType[...], L2]): ...

    # Same instructions, no wait; the caller calls wait_store / wait_load.
    @staticmethod
    def copy_from_async[...](...): ...
    @staticmethod
    def copy_to_async[...](...): ...
    @staticmethod
    def wait_store(): ...
    @staticmethod
    def wait_load(): ...
```

Constructing a tile from an allocation:

```mojo
comptime ACCUM_LAYOUT = TileLayout(
    shape=Coord(Idx[128], Idx[N]),
    stride=Coord(Idx[1], Idx[TMEM_NUM_LANES]),
)
comptime AccumTile = TileTensor[
    .float32, type_of(ACCUM_LAYOUT), MutAnyOrigin, Engine=TMemEngine
]
comptime AccumStorage = TMemStorage[.float32, MutAnyOrigin]

var accum = AccumTile(AccumStorage(alloc.addr), ACCUM_LAYOUT)
var warp_tile = accum.tile[32, N](Coord(Int(warp_id()), Idx[0]))
var my_row = warp_tile.tile[1, N](Coord(Idx[0], Idx[0]))  # same storage
var regs = stack_allocation[dtype = .float32](row_major[1, N]())
regs.copy_from(my_row)  # tcgen05.ld x N, one wait
my_row.copy_from(regs)  # tcgen05.st x N, one wait

# Several rows behind one wait.
tmem_copy_async(regs_a, row_a)
tmem_copy_async(regs_b, row_b)
TMemEngine.wait_load()
```

The storage struct's parameters cannot be inferred from a bare `UInt32`, so
callers name the storage type, as `AccumStorage` does above. The MMA still
needs the raw address for its descriptor, which is the handle's `addr`.

## Implementation plan

Each step has its own verification. All GPU steps run on a remote B200:

```bash
./bazelw test --config=remote-b200 //max/kernels/test/gpu/layout:test_tmem_engine.mojo.test
```

1. **Skeleton.** Done. `max/kernels/src/layout/tmem_engine.mojo`, exported
   from the package `__init__.mojo`; the build glob picks it up and the
   `//max:max_mojo` dependency for the `tcgen05` wrappers was already
   present. Host tests in `max/kernels/test/layout/test_tmem_engine.mojo`
   cover `offset()`, `distance()`, `unsafe_cast()`, and the quadrant address
   `tile()` produces.

2. **Codegen test.** Done, in
   `max/kernels/test/gpu/layout/test_tmem_engine.mojo`, modeled on
   `max/kernels/test/gpu/basics/test_tcgen05.mojo`. The round-trip kernel
   compiled for `sm_100a` emits `tcgen05.ld.sync.aligned.32x32b.x8.b32` and
   the matching `st`, each followed by its wait. The target is gated on
   `//:b200_gpu` with its own minimal-dependency rule. A later codegen test
   asserts a 128-column row is two `x64` instructions and two waits per
   direction, never an `x128`.

3. **Round-trip execution test.** Done for the M=128 flat layout across four
   warps and 128 lanes, and for the nested M=64 layout, both on a B200. Each
   thread copies a lane-and-column tagged register tile into its row view,
   copies it back out, and the host compares every cell.

4. **Batched copies.** Done. `copy_from()` and `copy_to()` move a row
   between TMEM and any pointer-backed tile with one instruction per
   power-of-two chunk and one wait per copy; the codegen test asserts a
   32-column row is a single `x32` instruction each way. The `_async`
   variants and `wait_store()` / `wait_load()` let several copies share a
   wait, and a second codegen test asserts two half-row copies produce two
   `x16` instructions and one wait. Comparing the emitted PTX with the
   `TmemTensor.load_fragments()` path in one epilogue is part of step 5.

   A register-budget test checks the cap against a real budget rather than
   against the emitted PTX: a kernel under launch bounds that hand `ptxas`
   128 registers per thread, the width the ISA would let one instruction
   name, copies a 128-column row from global memory through TMEM and back.
   The test compiles it on the device and asserts the function reports no
   local memory and stayed under the budget.

5. **Bridges and one consumer.** Add `as_tile_tensor()` to `TmemTensor` and
   `TMemTile`. Port one epilogue path, the non-transposed `output_writer`
   path or the attention softmax warp reading the S accumulator, and run the
   existing SM100 smoke tests. The epilogue port requires a `16x256b`
   engine.

6. **Fusion.** On the fusion graph branch, `GpuEngine.load_into()` already
   calls `tensor.load[width](coord)`, so TMEM leaves work once
   `materialize()` walks coordinates one thread per row. Row folds fit the
   same ownership, which is the online-softmax-over-S case.

## Testing notes

- Codegen assertions do not need execution but still target the current GPU,
  so they are gated on `//:b200_gpu` like `test_tcgen05.mojo`.
- The round-trip test is the only place the layout table and the engine's
  address encoding are checked against hardware. Tag every stored value with
  both lane and column so a wrong stride shows up as a transposition rather
  than a silent pass.
- Keep a test for the M=64 nested layout separate from M=128, because the
  two disagree exactly where the current `is_lower_required` logic lives.
- PTX never shows a spill; `ptxas` decides those. A register-pressure claim
  needs a kernel compiled on the device under launch bounds, with
  `LOCAL_SIZE_BYTES` and `NUM_REGS` read back from the function.

## Open questions

- **`cta_group=2` placement.** The per-CTA lane layout for two-CTA MMA is
  not derivable from the current code alone. The epilogue's
  `c_smem_tile_m = 32 if cta_group == 2` branch and the ISA data-path figures
  are the inputs; the round-trip test is the arbiter.
- **Two-byte element types.** `pack::16b` and a 32-bit reinterpretation are
  needed for `bfloat16` tiles, such as the P operand written back to TMEM in
  attention. Start with `float32`.
- **Chunk order.** A row whose length is not a power of two is split
  largest chunk first, so a 40-column row is one `x32` and one `x8`. Whether
  a different split matters for the hardware is unmeasured.
- **Higher-rank layouts.** A logically 3D or 4D tile has no agreed mapping
  onto TMEM's two physical dimensions, so the copy path requires rank 2.
  Trainium's SBUF and PSUM face the same question, and the answer should be
  shared across the non-flat engines rather than chosen here.
- **`tcgen05.cp`.** A bulk shared-memory to TMEM copy exists. A
  `copy_from()` overload from a shared-memory `TileTensor` could use it once
  the descriptor requirements are understood.
- **Same-engine gate on in-place ops.** Split-K read-modify-write goes
  through a register tile today and would continue to. If that proves too
  verbose, a `TensorOps` conformance that loads, computes, and stores per
  row with batched waits is possible but is not part of the initial scope.
