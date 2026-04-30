---
title: Mojo nightly
---

This version is still a work in progress.

## ✨ Highlights

## Documentation

- Compilation targets docs instructs how to inspect your current platform,
  select a target configuration, and generate code for that target. Use it to
  build for your own system or target other CPUs, operating systems, and
  accelerators.

- Mojo language reference covers lexical elements, expressions, statements,
  numeric types, struct declarations, trait declarations.

- Functions reference page improves discoverability of new function features.

- Split operators manual into separate pages; refreshed coverage and added
  tutorial, operator tests, and new reference page.

- Negative examples and errors added to reference pages highlight sharp
  edges of the language.

- MLIR reference page introduces inline MLIR to developers in Mojo code.

- Adds docs for non-nullable pointers and provides sample code showing
  how to use `Optional` with `UnsafePointer`.

## Language enhancements

- Added type refinement based on compile time assumptions, enabling Mojo to
  narrow types from `where` clauses, `comptime if` statements, and
  `comptime assert` statements. Refinements in a scope are driven by
  `conforms_to()` expressions.

  Before:

  ```mojo
  def __contains__(self, value: Self.T) -> Bool where conforms_to(Self.T, Equatable):
      for item in self:
          if trait_downcast[Equatable](item) == trait_downcast[Equatable](value):
              return True
      return False
  ```

  After:

  ```mojo
  def __contains__(self, value: Self.T) -> Bool where conforms_to(Self.T, Equatable):
      for item in self:
          if item == value:
              return True
      return False
  ```

- Improved diagnostics for onboarding-priority parser errors in Mojo
  for clarity and UX.

- Migrated monorepo from `fn` to using `def` for function declaration.
  Warned on use of `fn` and will deprecate `fn` in the next release.

- Updated signature error diagnostics and added related tests.

- Mojo now uses `NoneType` instead of an empty tuple to mark constructor
  using literals.

- The ternary `if/else` expression now coerces each element to its contextual
  type when it is obvious. For example, this works instead of producing an
  error about incompatible metatypes:

  ```mojo
    comptime some_type: Movable = Int if cond else String
  ```

- Unified closures now accept default capturing conventions when there are
  explicit captures already.

  ```mojo
  def captures_with_default_convention():
    var a, b, c, d = ("a", "b", "c", "d")
    def my_fn() unified {mut a, b, c^, read}:
      # capture:
      # `a` by mut reference
      # `b` by immut reference
      # `c` by moving
      # `d` by immut reference (the default 'read' convention)
      use(a, b, c, d)
  ```

- Added `abi("C")` as a function effect for declaring C calling convention on
  function definitions and function pointer types. Functions marked with
  `abi("C")` use the platform C ABI (System V x86-64 / ARM64 AAPCS) for
  struct arguments and return values, enabling safe interop with C libraries:

  ```mojo
  # C-ABI function definition (safe as a callback into C code)
  def add(a: Int32, b: Int32) abi("C") -> Int32:
      return a + b

  # C-ABI function pointer type (safe for use with DLHandle.get_function)
  var f = handle.get_function[def(Float64) abi("C") -> Float64]("sqrt")
  ```

  `DLHandle.get_function[]` now enforces that the type parameter carries
  `abi("C")`, preventing silent ABI mismatches when loading C symbols.

- String literals now support `\uXXXX` and `\UXXXXXXXX` unicode escape
  sequences, matching Python. The resulting code point is stored as UTF-8.
  Invalid code points and surrogates are rejected at parse time.

- Added support for conditional `RegisterPassable` conformance.

- Variadic lists and packs can be forwarded through runtime calls with `*pack`
  when the callee takes a compatible variadic list/pack.

  ```mojo
  def callee[*Ts: Writable](*args: *Ts):
      comptime for i in range(args.__len__()):
          print(args[i])

  def forwarder[*Ts: Writable](*args: *Ts):
      callee(*args)

  forwarder(1, "hello", 3.14)  # prints each value on a separate line
  ```

- Heterogenous variadic packs can now be specified with a `SomeType` helper
  function. These two are equivalent:

  ```mojo
  def foo[*arg_types: Copyable](*args: *arg_types) -> Int: ...
  def foo(*args: *SomeTypeList[Copyable]) -> Int: ...
  ```

- T-strings can now be used in `comptime assert` messages:

  ```mojo
    def foo[i: Int]():
        comptime assert i > 5, t"expected i > 5, got {i}"
  ```

## Language changes

- Variadic parameters lists are now passed instead of `ParameterList` and
  `TypeList` instead of `!kgen.param_list`. This makes it much more ergonomic to
  work with these types, e.g. simple logic just works:

  ```mojo
  def callee[*values: Int]():
      var v = 0
      for i in range(len(values)):
          v += values[i]
      for elt in values:
          v += elt
  ```

  Similarly, the `ParameterList`/`TypeList` structs have other methods for
  transforming the value list. As such, a variety of values from the `Variadic`
  struct have started moving over to being members of these types.

- All Mojo functions now has a unique "function literal type". In practice, it
  means that:

  ```mojo
  # type_of(foo) != type_of(bar)

  def foo(): pass
  def bar(): pass
  ```

- Mojo now warns on uses of the legacy `fn` keyword. Please move to `def` as
  this will upgrade to an error in the future.

- Import statements of the form `from pkg import ...` no longer make `pkg`
  available to the module.

## Library changes

- Removed explicit `trait_downcast`/`trait_downcast_var` across the standard
  library sources, now that Mojo applies type refinement from comptime
  assumptions. Public APIs are unchanged. Updated files:
  - `stdlib/std/builtin/`: `_stubs.mojo`, `bool.mojo`
  - `stdlib/std/collections/`: `deque.mojo`, `dict.mojo`, `inline_array.mojo`,
    `linked_list.mojo`, `list.mojo`, `optional.mojo`, `set.mojo`
  - `stdlib/std/iter/__init__.mojo`, `stdlib/std/itertools/itertools.mojo`
  - `stdlib/std/memory/`: `arc_pointer.mojo`, `owned_pointer.mojo`, `span.mojo`

- Consolidated the reflection APIs in `std.reflection` behind a unified entry
  point `reflect[T]()` returning a `Reflected[T]` handle. `reflect` is
  auto-imported via the prelude, so it is available without an explicit
  import. Methods on the handle replace the family of `struct_field_*` free
  functions (dropping the `struct_` prefix — only structs have fields) and
  the `get_type_name` / `get_base_type_name` free functions:

  ```mojo
  struct Point:
      var x: Int
      var y: Float64

  def main():
      comptime r = reflect[Point]()
      print(r.name())                          # "Point"
      print(r.field_count())                   # 2
      print(r.field_names()[0])                # x
      comptime y_type = r.field_type["y"]()    # Reflected[Float64]
      print(y_type.name())                     # "SIMD[DType.float64, 1]"
      print(reflect[List[Int]]().base_name())  # "List"
      var v: y_type.T = 3.14
  ```

  Methods on `Reflected[T]`: `name[qualified_builtins=]`, `base_name`,
  `is_struct`, `field_count`, `field_names`, `field_types`,
  `field_index[name]`, `field_type[name]`,
  `field_offset[name=]/[index=]`, and `field_ref[idx](s)`. The
  `field_type[name]()` method returns a `Reflected[FieldT]`, so reflection
  is fully composable.

  The legacy free functions — `struct_field_count`, `struct_field_names`,
  `struct_field_types`, `struct_field_index_by_name`,
  `struct_field_type_by_name`, `struct_field_ref`, `is_struct_type`,
  `offset_of`, `get_type_name`, `get_base_type_name` — and the
  `ReflectedType[T]` wrapper are now `@deprecated` and delegate to the new
  API. They will be removed in a future release.

- Added `struct_field_ref[idx, T](ref s)` to `std.reflection` for accessing
  struct fields by index without copying. The function returns a reference
  with the same mutability as `s` and works with both concrete and generic
  struct types, including parametric indices in `comptime for` loops. The
  default implementations of `Hashable`, `Equatable`, and `Writable` now use
  this library function instead of the `__struct_field_ref` magic.

- The `Boolable`, `Defaultable`, and `Writable` traits no longer inherit from
  `ImplicitlyDestructible`. Generic code that relied on receiving the
  destructor bound transitively through these traits must now spell it out
  explicitly, for example `T: Writable & ImplicitlyDestructible`.

- The `Variadic` suite of low-level operation has been refactored and migrated
  to being members of the `TypeList` and `ParameterList` types, making them more
  ergonomic to work with and more accessible.

- Atomic operations have moved to a dedicated `std.atomic` module. The
  `Consistency` type has been renamed to `Ordering` and its `MONOTONIC`
  member has been renamed to `RELAXED` to align with conventions used by
  other languages. Update existing code as follows:

  ```mojo
  # Before
  from std.os import Atomic
  from std.os.atomic import Atomic, Consistency, fence

  _ = atom.load[ordering=Consistency.MONOTONIC]()

  # After
  from std.atomic import Atomic, Ordering, fence

  _ = atom.load[ordering=Ordering.RELAXED]()
  ```

- Added `Optional.destroy_with(destroy_func)`, which destroys an `Optional[T]`
  in-place using a caller-provided destructor. This enables `Optional` to hold
  element types that are not `ImplicitlyDestructible` (for example, types
  marked `@explicit_destroy`), mirroring `Variant.destroy_with`. Both
  `Optional.destroy_with` and `Variant.destroy_with` now accept closures that
  capture local state in addition to plain function references. Note that
  `Variant.destroy_with` callers must now pass the destroyed type explicitly
  (for example, `v^.destroy_with[Int](destroy_func)`) since `T` can no longer
  be inferred from the closure type.

- `assert_raises` now catches custom `Writable` error types, not just `Error`.

- Added UAX #29 grapheme cluster segmentation to `String` and `StringSlice`.
  New APIs: `graphemes()` returns a `GraphemeSliceIter` that yields each
  user-perceived "character" as a `StringSlice`, and `count_graphemes()` returns
  the grapheme cluster count. This correctly handles combining marks, emoji ZWJ
  sequences, flag emoji, Hangul syllables, and other multi-codepoint clusters.

- `StringSlice` now supports slicing by grapheme cluster via the `grapheme=`
  keyword argument, mirroring the existing `byte=` indexer. For example,
  `s[grapheme=0:3]` returns a `StringSlice` covering the first three grapheme
  clusters, and `s[grapheme=i:i+1]` extracts the *i*-th grapheme. Out-of-range
  ends are clamped to the end of the string; negative indices are not supported.
  Because grapheme boundaries are discovered by a forward scan, this operation
  is O(n) in the byte length — prefer `byte=` slicing when you already have
  byte offsets.

- `GraphemeSliceIter` now supports reverse iteration. `next_back()` and
  `peek_back()` return the last grapheme cluster in the remaining range, and
  `StringSlice.graphemes_reversed()` / `String.graphemes_reversed()` return a
  `GraphemeSliceIter` whose `for`-loop iteration walks clusters from end to
  start. `next()` and `next_back()` can be interleaved on the same iterator.
  Reverse iteration costs more per cluster than forward iteration because the
  UAX #29 state machine is inherently forward-scanning: `next_back()` backs
  up to a guaranteed grapheme boundary (the start of the string or a
  Control/CR/LF codepoint) and rescans forward. The safe boundary is cached
  across reverse calls — a forward `next()` invalidates it — so per-call cost
  is dominated by forward-scan length: small in text containing line breaks
  or whitespace, growing with the distance back to such a codepoint in long
  runs without them.

- Added grapheme-aware algorithms on `String` and `StringSlice`:
  - `grapheme_indices()` returns a `GraphemeIndicesIter` that yields
    `(byte_offset, grapheme)` pairs, mirroring Rust's
    `str::grapheme_indices`. Useful for text editors or UIs that need to
    map cursor byte positions back to grapheme boundaries.
  - `nth_grapheme(n)` returns the `n`-th grapheme cluster as an
    `Optional[StringSlice]`, or `None` when `n` is out of range.
  - `split_at_grapheme(n)` returns `Tuple[StringSlice, StringSlice]`
    holding the prefix `[0, n)` and suffix `[n, count)` of grapheme
    clusters in a single pass, clamping `n` to the total count.

- `count_graphemes()` now takes a fast path over runs of printable ASCII
  (U+0020..U+007E). Each such byte has GBP `Other` and two consecutive
  safe-ASCII bytes always have a grapheme-cluster break between them
  (GB999), so safe-ASCII runs can be counted at one grapheme per byte
  without entering the UAX #29 state machine. On pure-ASCII text this is
  roughly 10x faster (~0.38 ms vs. ~3.85 ms for 1 MB of English), and
  ~5-6x faster on ASCII-dominant mixed text (Spanish UN charter). Pure
  non-ASCII text (Arabic, Russian, Chinese) is unchanged.

- Variadics of types have been moved to the `TypeList` struct.
  One can write operations such as:

  ```mojo
  comptime assert TypeList[Trait=AnyType, Int, String]().contains[Bool]
  ```

- `abort(message)` now includes the call site location in its output. The
  location is automatically captured and printed alongside the message. You can
  also pass an explicit `SourceLocation` to override it:

  ```mojo
  abort("something went wrong")
  # prints: ABORT: path/to/file.mojo:42:5: something went wrong

  var loc = current_location()
  abort("something went wrong", location=loc)
  ```

- `abort(message)` now prints its message on Nvidia and AMDGPU, including
  block and thread IDs. Previously, the message was silently suppressed on
  these GPUs. On Apple GPU, the message is silently suppressed for now.

- `SourceLocation` fields (`line`, `col`, `file_name`) are now private.
  Use the new accessor methods `line()`, `column()`, and `file_name()` instead.

- Fixed default alignment in `TileTensor.load()` and `TileTensor.store()` to
  use the caller-specified `width` parameter instead of `Self.element_size`.

- Added uninitialized memory read detection for float loads. When compiled
  with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`, every float load is checked
  against the debug allocator's poison patterns (0xFF host fill and canonical
  qNaN device fill). A match triggers `abort()` with a descriptive message.
  When disabled (the default), zero runtime overhead. For MAX pipelines, set
  `MODULAR_MAX_DEBUG_UNINITIALIZED_READ_CHECK=true` (or the
  `max-debug.uninitialized-read-check` config key, or
  `InferenceSession.debug.uninitialized_read_check = True`) to enable both the
  debug allocator and the load-time checks automatically.

- Added `CompilationTarget.is_apple_m5()` to `std.sys` for detecting Apple M5
  targets at compile time. `is_apple_silicon()` now includes M5 in its check.

- Added Apple M5 MMA intrinsics (`apple_mma_load`, `apple_mma_store`,
  `_mma_apple`) in `std.gpu.compute.arch.mma_apple`, enabling hardware matrix
  multiply-accumulate on Apple GPU.

- Standard library types now use conditional conformances, replacing previous
  `_constrained_conforms_to` checks:
  - `Span`: `Writable`, `Hashable`
  - `Tuple`, `Optional`, `Variant`, and `UnsafeMaybeUninit`: `RegisterPassable`
  - `Variant`: `Copyable`, `ImplicitlyCopyable`

- `Tuple` now conditionally conforms to `Defaultable`, so generic
  `T: Defaultable` code can default-construct tuples when all element types are
  `Defaultable`.

- `OwnedDLHandle.get_symbol()` now returns `Optional[UnsafePointer[...]]`
  instead of aborting when a symbol is not found. This allows callers to handle
  missing symbols gracefully.

- `UnsafePointer` is now non-null by design. See the
  [non-null pointer proposal](https://github.com/modular/modular/blob/main/mojo/proposals/non-null-pointer.md)
  for the full design and migration timeline.

  The default null constructor `__init__(out self)` and `__bool__(self)` method
  are now deprecated, and `UnsafePointer` no longer conforms to `Defaultable` or
  `Boolable`.

  To migrate, express nullability explicitly with
  `Optional[UnsafePointer[...]]`, which has the same layout as `UnsafePointer`
  (the null address is the `None` niche) so nullable pointers remain
  zero-overhead and can be used across C-FFIs.

  ```mojo
  # Before: null default construction
  var ptr = UnsafePointer[Int, origin]()

  # After: express absence with Optional
  var ptr: Optional[UnsafePointer[Int, origin]] = None

  # Before: Bool-based null check
  if ptr:
      use(ptr[])

  # After: check the Optional, then unwrap
  if ptr:
      use(ptr.value()[])
  ```

  If you specifically need a non-null placeholder for a field that will be
  populated later (for example, a buffer that is allocated on demand) use
  `UnsafePointer.unsafe_dangling()`, which returns a well-aligned but dangling
  pointer. Note that `unsafe_dangling()` is not a null sentinel: types that
  lazily allocate must track initialization separately.

- GPU primitive id accessors (e.g. `thread_idx`) have migrated from `UInt` to
  `Int`.

  This is part of a broader migration to standardize on the `Int` type for all
  sizes and offsets in Mojo.

  To provide a gradual migration path, explicitly typed aliases are
  available temporarily.

  | Base         | `UInt` Accessor   | `Int` Accessor    |
  |--------------|-------------------|-------------------|
  | `thread_idx` | `thread_idx_uint` | `thread_idx_int`  |
  | `thread_dim` | `thread_dim_uint` | `thread_dim_int`  |
  | `block_dim`  | `block_dim_uint`  | `block_dim_int`   |
  | `grid_dim`   | `grid_dim_uint`   | `grid_dim_int`    |
  | `global_idx` | `global_idx_uint` | `global_idx_int`  |
  | `lane_id`    | `lane_id_uint`    | `lane_id_int`     |
  | `warp_id`    | `warp_id_uint`    | `warp_id_int`     |

  Code can preserve its prior behavior by using a renaming import of the
  `thread_idx_uint` alias:

  ```diff
  - from std.gpu import thread_idx
  + from std.gpu import thread_idx_uint as thread_idx
  ```

  Note that `thread_idx_uint` and the other `_*uint` aliases will eventually
  be deprecated and removed as well.

  After a temporary deprecation acting as a "speed bump" in the 2026-03-29
  nightly release, `thread_idx` etc. have changed from `UInt` to `Int`.

  Code built with a version where `thread_idx` is still `UInt`, can proactively
  migrate to the eventual `Int` behavior using the `thread_idx_int` alias:

  ```diff
  - from std.gpu import thread_idx
  + from std.gpu import thread_idx_int as thread_idx

  # ... update file to reflect change from `UInt` to `Int` ...
  ```

- Added `IterableOwned` trait to the iteration module. Types conforming to
  `IterableOwned` implement `__iter__(var self)`, which consumes the collection
  and returns an iterator that owns the underlying elements.
  - `List` now conforms to `IterableOwned`.
  - `Optional` now conforms to `IterableOwned`.
  - `Deque` now conforms to `IterableOwned`.
  - `LinkedList` now conforms to `IterableOwned`.
  - `Dict` now conforms to `IterableOwned`.
  - `Set` now conforms to `IterableOwned`.
  - `Counter` now conforms to `IterableOwned`.
  - `InlineArray` now conforms to `IterableOwned`.
  - `Span` now conforms to `IterableOwned` (conditional on `T: Copyable`).
    The owned iterator yields copies of elements by value.
  - Iterator adaptors (`enumerate`, `zip`, `map`, `peekable`, `take_while`,
    `drop_while`, `product`, `cycle`, `count`, `repeat`) now conform to
    `IterableOwned`.
  - Added owned overloads of `enumerate()`, `zip()`, `map()`, `peekable()`,
    `take_while()`, `drop_while()`, `product()`, and `cycle()` that consume the
    input iterable.

- `CStringSlice` can no longer represent a null pointer. To represent
  nullability use `Optional[CStringSlice]` which is guaranteed to have the same
  size and layout as `const char*`, where `NULL` is the empty `Optional`.

- `external_call`'s `return_type`'s requirements has been relaxed from
  `TrivialRegisterPassable` to `RegisterPassable`.

- Negative indexing on all stdlib collections has been removed to enable cheap
  CPU bounds checks by default:
  - `List`
  - `Span`
  - `InlineArray`
  - `String`
  - `StringSlice`
  - `LinkedList`
  - `Deque`
  - `IntTuple`

  Using a negative `IntLiteral` for indexing will now trigger a compile-time
  error, for example:

  ```text
  /tmp/main.mojo:3:12: note: call expansion failed with parameter value(s): (..., ...)
          print(x[-1])
              ^
  constraint failed: negative indexing is not supported, use e.g. `x[len(x) - 1]` instead
  ```

  Update any `x[-1]` to `x[len(x) - 1]`, following the compiler errors to
  your call sites as above.

  This does not affect any MAX ops that support negative indexing.

- Bounds checking is now on by default for all collections on CPU, and will show
  you the call site in your code where you triggered the out of bounds access:

  ```mojo
  def main():
      var x = [1, 2, 3]
      print(x[3])
  ```

  ```text
  At: /tmp/main.mojo:3:12: Assert Error: index 3 is out of bounds, valid range is 0 to 2
  ```

  Bounds checking is still off by default for GPU to avoid performance
  penalties. To enable it for tests:

  ```bash
  mojo build -D ASSERT=all main.mojo
  ```

  To turn off all asserts, including CPU bounds checking:

  ```bash
  mojo build -D ASSERT=none main.mojo
  ```

- `alloc[T](count, alignment)` will now `abort` if the underlying allocation
  failed.

- Added `Variadic.contains_value` comptime alias to check whether a variadic
  sequence contains a specific value at compile time.

- `ArcPointer` now conditionally conforms to `Hashable` and `Equatable` when
  its inner type `T` does. Both `__eq__` and `__hash__` delegate to the managed
  value, matching C++ `shared_ptr` and Rust `Arc` semantics. This makes
  `ArcPointer` usable as a `Dict` key or `Set` element with value-based
  equality. Pointer identity is still available via the `is` operator.

- `Path` now conforms to `Comparable`, enabling lexicographic ordering and use
  with `sort()`.

- `range()` overloads that took differently-typed arguments or arguments that
  were `Intable`/`IntableRaising` but not `Indexer` have been removed. Callers
  should ensure they're passing consistent integral argument types when calling
  `range()`.

- `Consistency` now has a default constructor that selects `RELEASE` ordering on
  Apple GPU and `SEQUENTIAL` on all other targets. All `Atomic` methods and
  `fence` use this platform-aware default instead of hard-coding `SEQUENTIAL`.

- `NDBuffer` has been fully removed. Please migrate to `TileTensor`.

- Added a generic `__contains__` method to `Span` for any element type
  conforming to `Equatable`, not just `Scalar` types.

- Fixed `blocked_product` in `tile_layout` to zip block and tiler dimensions
  per mode, matching the legacy `blocked_product` behavior.

- Added `Span`-based overloads for `enqueue_copy`, `enqueue_copy_from`, and
  `enqueue_copy_to` on `DeviceContext`, `DeviceBuffer`, and `HostBuffer`,
  providing a safer alternative to raw `UnsafePointer` for host-device memory
  transfers.

- `String.__len__()` has been deprecated. Prefer to use `String.byte_length()`
  or `String.count_codepoints()`.

- Added `map()` and `and_then()` methods to `Optional`. `map()` transforms
  the contained value by applying a function, returning `Optional[To]`.
  `and_then()` chains operations that themselves return an `Optional`, enabling
  flat-mapping over fallible computations.

  ```mojo
  var o = Optional[Int](42)

  def closure(n: Int) unified {} -> String:
    return String(n + 1)

  var mapped: Optional[String] = o.map[To=String](closure)
  print(mapped) # Optional("43")
  ```

- Added `std.memory.forget_deinit()` to enable low-level code to skip the usual
  requirement to run a destructor for a value. This function should be used
  rarely, when building low-level abstractions.

- `parallelize`, `parallelize_over_rows` (in
  `std.algorithm.backend.cpu.parallelize`) and the `elementwise` overloads in
  `std.algorithm.functional` now accept an optional trailing
  `ctx: Optional[DeviceContext] = None` parameter. When supplied, the provided
  CPU `DeviceContext` is forwarded to `sync_parallelize` so that parallel work
  runs on that context; when omitted, the previous behavior is preserved. This
  is a step toward running CPU ops on specific NUMA nodes.

## Tooling changes

- The Mojo debugger now shows a `Variant` variable's active type name and
  value in LLDB — e.g. `Int(42)` or `String("hello")` — instead of exposing
  raw `_DefaultVariantStorage` internals.

- The Mojo debugger now displays scalar types (e.g. `UInt8`, `Float32`) as
  plain values instead of `([0] = value)`, and elides internal `_mlir_value`
  wrapper fields from struct display.

- `mojo format` no longer supports the deprecated `fn` keyword, nor the
  removed `owned` argument convention.

- Comptime function calls now print more nicely in error messages and generated
  documentation, not including `VariadicList`/`VariadicPack` and including
  keyword argument labels when required.

## GPU programming

- Added support for AMD MI250X accelerators.

## ❌ Removed

- The `escaping` function effect is no longer supported. Migrate
  `def(...) escaping -> T` closures to `unified` closures.

- The deprecated `@doc_private` decorator has been removed. Use `@doc_hidden`
  instead.

- Removed the `store_release`, `store_relaxed`, `load_acquire`, and
  `load_relaxed` helpers from `std.gpu.intrinsics`. Use
  [`Atomic[dtype, scope=...].store`](/mojo/std/atomic/atomic/Atomic/#store) and
  [`Atomic[dtype, scope=...].load`](/mojo/std/atomic/atomic/Atomic/#load) with
  the desired [`Ordering`](/mojo/std/atomic/atomic/Ordering/) instead:

  ```mojo
  # Before
  from std.gpu.intrinsics import store_release, load_acquire
  store_release[scope=Scope.GPU](ptr, value)
  var v = load_acquire[scope=Scope.GPU](ptr)

  # After
  from std.atomic import Atomic, Ordering
  Atomic[dtype, scope="device"].store[ordering=Ordering.RELEASE](ptr, value)
  var v = Atomic[dtype, scope="device"].load[ordering=Ordering.ACQUIRE](ptr)
  ```

## 🛠️ Fixed

- Fixed `math.sqrt` on `Float64` on NVIDIA GPU producing a cryptic
  `could not find LLVM intrinsic: "llvm.nvvm.sqrt.approx.d"` failure at LLVM
  IR translation time. `math.sqrt` now rejects `Float64` on NVIDIA GPU at
  compile time with the message `DType.float64 isn't supported for approx
  sqrt on NVIDIA GPU`. The existing `math.sin` and `math.cos` constraint
  messages were also sharpened to name the op (`DType.float64 isn't supported
  for sin/cos on NVIDIA GPU`).
  ([Issue #6434](https://github.com/modular/modular/issues/6434))

- Fixed pack inference failing with `could not infer type of parameter pack ...
  given value with unresolved type` when passing list, dict, set, or slice
  literals to a `*Ts`-bound variadic pack parameter (e.g.
  `def foo[*Ts: Iterable](*args: *Ts)`). Pack inference now applies the same
  default-type fallback that single-argument trait-bound parameters already
  use, so `foo([1, 2, 3], [4, 5, 6])` resolves each literal to its default
  type (e.g. `List[Int]`) before binding the pack.

- Fixed `mojo` aborting at startup with `std::filesystem::filesystem_error`
  when `$HOME` is not traversable by the running UID (common in containerized
  CI where the image's build-time UID differs from the runtime UID). The
  config search now treats permission errors as "not found" and falls through
  to the next candidate.
  ([Issue #6412](https://github.com/modular/modular/issues/6412))

- `mojo run` and `mojo debug` now honor `-Xlinker` flags by loading the
  referenced shared libraries into the in-process JIT. Previously the flags
  were dropped (with a `-Xlinker argument unused` warning), leaving programs
  that called into external shared libraries via `external_call` unable to
  resolve those symbols at runtime (so `mojo build` worked but `mojo run` did
  not). The supported forms mirror what the system linker accepts: `-Xlinker
  -L<dir>`, `-Xlinker -l<name>`, `-Xlinker -rpath <dir>`, and `-Xlinker
  <absolute-path-to-shared-library>`. Flags that have no meaning under JIT
  are reported as a warning and ignored.
  ([Issue #6155](https://github.com/modular/modular/issues/6155))

- Fixed `libpython` auto-discovery failing for Python 3.14 free-threaded builds.
  The discovery script constructed the library filename without the ABI flags
  suffix (e.g. looked for `libpython3.14.dylib` instead of
  `libpython3.14t.dylib`).
  ([Issue #6366](https://github.com/modular/modular/issues/6366))
- Fixed `RTLD.LOCAL` having the wrong value on Linux. It was set to `4`
  (`RTLD_NOLOAD`) instead of `0`, causing `dlopen` with `RTLD.NOW | RTLD.LOCAL`
  to fail. ([Issue #6410](https://github.com/modular/modular/issues/6410))

- Fixed `mojo format` crashing after upgrading Mojo versions due to a stale
  grammar cache. ([Issue #6144](https://github.com/modular/modular/issues/6144))

- Fixed `atof` producing incorrect results for floats near the
  normal/subnormal boundary (e.g., `Float64("4.4501363245856945e-308")`
  returned half the correct value).
  ([#6196](https://github.com/modular/modular/issues/6196))

- [Issue #5872](https://github.com/modular/modular/issues/5872): Fixed a
  compiler crash ("'get_type_name' requires a concrete type") when using
  default `Writable`, `Equatable`, or `Hashable` implementations on structs
  with MLIR-type fields (e.g. `__mlir_type.index`). The compiler now correctly
  reports that the field does not implement the required trait.

- Fixed `Atomic.store` silently dropping the requested `scope`. The previous
  implementation lowered to `atomicrmw xchg` without forwarding `syncscope`,
  so `Atomic[..., scope="device"].store(...)` was emitting a system-scope
  store on NVPTX (extra L2/NVLink fences) and an over-synchronized store on
  AMDGPU. `Atomic.store` now lowers via `pop.store atomic syncscope(...)`,
  emitting `st.release.<scope>` on NVPTX and a properly-scoped LLVM atomic
  store on AMDGPU. The Mojo API surface is unchanged.

- Fixed `Process.run()` not inheriting the parent's environment variables.
  Child processes spawned via `Process.run()` now correctly receive the
  parent's environment.

- Fixed `\xhh` and `\ooo` escape sequences in string literals being
  interpreted as raw bytes instead of Unicode code points, which produced
  malformed UTF-8 for values `>= 0x80`. The escapes now match Python `str`
  semantics (and the existing `\u`/`\U` handling): `"\x85"` encodes U+0085
  (NEL) as two UTF-8 bytes and `ord("\x85")` returns `133` instead of `5`.
  Code that relied on `\xhh` to emit a single raw byte for non-ASCII values
  must construct the bytes explicitly (for example via a `List[Byte]`
  literal).
  ([Issue #2842](https://github.com/modular/modular/issues/2842))

- Fixed incorrect data layout for `MI250X` AMDGPU architectures.
  ([Issue #6451](https://github.com/modular/modular/issues/6451)
