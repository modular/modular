---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- Mojo now support `==` and `!=` for type equality check, and `_type_is_eq` is
  removed.

- Mojo now infers `Trait` for `TypeList.of` such that

  ```mojo
  comptime TL = TypeList.of[Int, Bool]
  # works without
  comptime TL = TypeList.of[Trait = AnyType, Int, Bool]
  ```

- Mojo now warns about redundant trait composition

  ```mojo
  # Warning: Redundant trait composition: 'Copyable' already implies 'AnyType'
  comptime T : AnyType & Copyable = xxx
  ```

- Keyword variadic arguments can now be forwarded to another function that takes
  keyword variadics, using Python style `**` syntax:

  ```mojo
  def takes_them(**kwargs: Int): ...
  def pass_them(**kwargs: Int):
    takes_them(**kwargs^)
  ```

- Struct fields are no longer allowed to hide `UnsafeAnyOrigin` within a
  struct, e.g. this is no longer accepted:

  ```mojo
  struct Example:
    # error: cannot use UnsafeAnyOrigin in a struct field.
    var ptr: UnsafePointer[Int, MutUnsafeAnyOrigin]
  ```

  This is because Mojo doesn't know that uses of `Example` contain an
  `UnsafeAnyOrigin` and therefore doesn't do lifetime extension for values in
  its context. The typical solution for this is to add an `Origin` parameter but
  you can also use `UntrackedOrigin` if you explicitly manage the lifetime of
  the underlying data:

  ```mojo
  struct Example[origin: Origin]:
    var ptr: UnsafePointer[Int, Self.origin]

  # OR

  struct Example:
    var ptr: UnsafePointer[Int, MutUntrackedOrigin]
  ```

  As a temporary workaround, you can decorate fields with
  `@__allow_legacy_any_origin_fields` to ignore the compiler error, however this
  decorator is not stable and will eventually be removed.

- Import resolution behavior has been made consistent. When resolving an import
  of a module or package, in any given directory the resolution in order of
  preference is: source packages; precompiled `.mojoc` files; source modules;
  legacy precompiled `.mojopkg` files.

  Previously the behavior was unspecified and would pick whichever matching
  name it found in the directory first.

## Language changes

- Relative imports must now use `from` (`from . import foo`); the `import .foo`
  form is no longer accepted.

- Absolute imports `import a.b.c` now bind all of `a`, `a.b`, and `a.b.c` into
  the scope, where previously only `a.b.c` was made available.

- A bug in import handling has been fixed where absolute imports of a package
  followed by an import of one of its submodules no longer result in a compiler
  error.

  ```mojo
  import a
  import a.b  # fixed; was: "invalid redefinition of 'a'"
  ```

- A bug in function-scoped imports has been fixed, allowing dotted imports:

  ```mojo
  def foo():
    import a.b

    a.b.foo() # fixed; was: "use of unknown declaration 'a'"
  ```

  Note that this was already working correctly for other forms of import
  (`import a`, `from a import b`, `from a.b import c`, etc).

- An imported package's submodules are now only accessible when the package's
  `__init__.mojo` re-exports those submodules.

  ```mojo
  import pkg

  # only ok if pkg/__init__.mojo re-exports 'sub'.
  # Re-export submodules with, e.g.,
  #   from . import sub
  # Use relative imports to avoid importing system packages.
  pkg.sub.foo()
  ```

  Note that absolute imports can always bring in that submodule, bypassing the
  `__init__.mojo`:

  ```mojo
  # always ok, regardless of the package's __init__.mojo
  import pkg.submodule

  pkg.submodule.foo()
  ```

- `where` clauses inside a parameter list (for example,
  `[x: Int where x > 0]`) are no longer supported, following a period of
  deprecation. Use a trailing `where` clause after the signature instead:

  ```mojo
  # Old (no longer supported):
  # fn foo[x: Int where x > 0]():

  # New:
  fn foo[x: Int]() where x > 0:
      pass
  ```

## Library changes

- `Int` is now an alias for `Scalar[DType.int]` and integer literals materialize
  to this `Scalar` type. Because of this some conversions have become more
  strict.

  A new `SIMDSize` type has been added for the width of `SIMD` itself and must
  be used when inferring a parameter based on a SIMD argument like so:

  ```mojo
  def frob[w: SIMDSize](v: SIMD[DType.int, w]): ...
  ```

  Alternatively the width can be unbound if you simply want to be parametric
  over any `SIMD` type:

  ```mojo
  def frob(v: SIMD[DType.int, _])
  ```

  The new `Int` should still be used in all other situations.

- `ImplicitlyDestructible` has been renamed to `ImplicitlyDeletable`, for better
  name consistency with its required `__del__()` "delete" special method.

- The `Reflected.field_type[name]` reflection member has been renamed to
  `Reflected.field[name]`, because it returns a chainable `Reflected` handle
  for the named field rather than the field's bare type, so the old name was
  not accurate. Retrieve the field's type from the handle's `.T` member, as in
  `reflect[T].field["x"].T`. Update call sites such as
  `reflect[T].field_type["x"]` to `reflect[T].field["x"]`.

- Several collection types now *conditionally* conform to `ImplicitlyDeletable`,
  conforming only when their element type does. This lets a collection hold
  non-`ImplicitlyDeletable` elements at all (previously such a collection failed
  to compile); a collection of non-deletable elements is itself linear and must
  be drained explicitly with the new `destroy_with()` method, which calls a
  closure on each element:

  ```mojo
  collection^.destroy_with(my_destroy_closure)
  ```

  Generic code that takes one of these collections by value may now need
  `& ImplicitlyDeletable` added to its element bound so the collection can be
  dropped:

  ```mojo
  def foo[T: Movable & ImplicitlyDeletable, //](var arr: InlineArray[T, 3]):
      pass
  ```

  Affected types:

  - `InlineArray[ElementType, size]`.
  - `Deque[ElementType]`
    - Element-destroying operations (`append`, `appendleft`, `extend`,
      `extendleft`, `insert`, `clear`, `remove`, etc.) still require
      `ElementType` to be `ImplicitlyDeletable`.
    - Consuming iteration (`for x in deque^`, the `IterableOwned` conformance)
      is likewise conditional, requiring `ElementType` to be
      `ImplicitlyDeletable`; generic code bounded on `IterableOwned` now rejects
      a non-conforming element type at the bound rather than failing later
      inside `__iter__()`. For deletable element types (the common case) this is
      transparent.

- Is is now possible to iterate over owned elements in
  `List`, `Dict`, `InlineArray`, `LinkedList`, and `Set`
  when the element type is not `Copyable`:

  ```mojo
  def iterate[T: Movable](var list: List[T]):
    # Consume elements
    for var x in list^:
        pass
  ```

  The `IterableOwned` conformance on several collections is now conditional
  on the element type conforming to `Movable & ImplicitlyDeletable`, dropping
  `Copyable`.

  Additionally, generic code bounded on `IterableOwned` now rejects a collection
  of non-conforming elements at the bound, rather than failing later inside
  `__iter__()`.

- The implicit conversion constructors that cast an `UnsafePointer` to
  `MutUnsafeAnyOrigin` or `ImmutUnsafeAnyOrigin` are now deprecated and emit a
  deprecation warning when used. `UnsafeAnyOrigin` is an unsafe escape hatch
  that silently extends unrelated lifetimes and disables exclusivity checking,
  so it should never be applied implicitly. Prefer keeping a concrete origin;
  if you must discard it, make the cast explicit with the
  `as_unsafe_any_origin()` method.

- Added `reflect[T].field_at[idx]` to the reflection API, the by-index dual
  of `reflect[T].field[name]`. It returns the reflection handle for the
  type of the field at `idx`, so a field's concrete type can be recovered while
  iterating fields by index (where the name is not available as a literal):

  ```mojo
  comptime y_type = reflect[Point].field_at[1]
  var v: y_type.T = 3.14  # y_type.T is the concrete field type
  ```

- Removed the implicit constructors that converted an `UnsafePointer` into an
  `Optional[UnsafePointer[..., UnsafeAnyOrigin]]`. Constructing an
  `Optional[UnsafePointer]` now preserves the pointer's real origin instead of
  silently widening it to `UnsafeAnyOrigin`. Two call-site updates may be
  needed:

  - Passing a concrete pointer where the parameter's origin is a genuinely
    fixed `MutAnyOrigin`/`ImmutAnyOrigin` (typically C-FFI signatures) now
    requires an explicit `as_unsafe_any_origin()`.

  - Because origins are now preserved, exclusivity checking applies to
    `memcpy()` (and similar) calls whose `dest` and `src` derive from the same
    buffer. An intra-buffer copy that previously compiled now errors with
    "argument of 'memcpy' call allows writing a memory location previously
    writable through another aliased argument". Opt out by making one argument
    an unsafe any-origin (the non-overlap of `dest` and `src` is already a
    `memcpy()` precondition):

    ```mojo
    memcpy(
        dest=buf + dst_off,
        src=(buf + src_off).as_unsafe_any_origin(),
        count=n,
    )
    ```

- The traits `ImplicitlyDeletable`, `Movable`, `Copyable`, and
  `ImplicitlyCopyable` are now stable.

- Removed `trait_downcast_var()`. Improvements to type refinement based on
  `where conforms_to(..)` and `comptime assert conforms_to(..)` make explicit
  value trait downcasting no longer necessary.

- Added `raise_python_exception()` to `std.python.bindings`, which translates a
  Mojo `Error` into a Python exception via `PyErr_SetString` and returns a null
  `PyObjectPtr`.

- Iterating over a `String`, `StringSlice`, or `StringLiteral` now yields
  grapheme clusters by default. Their `__iter__()` and `__reversed__()` methods
  return a `GraphemeSliceIter`, so `for c in my_string:` produces what a user
  perceives as a single "character" on screen. The lower-level views remain
  available when you want them: `codepoints()` or `codepoint_slices()` for
  Unicode scalars, and `bytes()` for raw UTF-8 bytes.

- `SIMD[DType.bool, N]` now has two new methods:
  - `first_true()` -- returns the index of the first `True` lane, or a
    caller-provided `default` (`-1` if unspecified) if all lanes are `False`.
    Replaces the manual `pack_bits` + `count_trailing_zeros` pattern.
  - `count_true()` -- returns the number of `True` lanes. A bool-specific
    alias for `reduce_bit_count()`.

## Tooling changes

- Added a `--lld-path` CLI flag. This overrides the LLD path that Mojo uses.

## GPU programming

- `DeviceContext.load_function` now keys its runtime cache on the requested
  entry-point name as well as the blob. Loading two different entry points
  (for example `kernel_a` and `kernel_b`) from a single PTX/cubin blob no
  longer collides — previously the second load silently returned the function
  resolved by the first. The cache also no longer keys on the entire blob
  when no module name is supplied: it keys on a short hash of the blob instead,
  so each call avoids copying, hashing, and byte-comparing the whole blob (and
  retaining a duplicate of it). The win scales with blob size and matters most
  for large multi-entry blobs loaded on the per-execution path.

- The `DeviceStream` type is now included in the API reference documentation.
  Returned by `DeviceContext.create_stream()` and
  `DeviceContext.create_external_stream()`, it provides methods for
  synchronizing and sequencing asynchronous GPU work (for example,
  `synchronize()`, `record_event()`, and `enqueue_wait_for()`). The type was
  already public but was previously hidden from the generated docs.

- Added an 8x8 `simdgroup_matrix` matrix multiply-accumulate primitive
  (`_mma_apple_8x8()`) with `apple_mma_load_8x8()` / `apple_mma_store_8x8()`
  fragment helpers for Apple Silicon GPUs in `std.gpu.compute.arch`. Unlike
  the 16x16 path (Apple M5 only), the 8x8 primitive is available on all Apple
  GPU generations (M1-M5). It accepts `Float16`, `BFloat16`, and `Float32`
  inputs with a `Float32` accumulator.

- Apple M5 `simdgroup_matrix` MMA now accepts FP8 (`float8_e4m3fn`,
  `float8_e5m2`) inputs with an F32 accumulator, alongside the existing
  F16/BF16/F32 and 8-bit integer types.

- Added `warp.match_any()`, which returns, for each warp lane, the mask of
  lanes whose value has the same bits. It uses NVIDIA's `match.any.sync`
  instruction, a `readfirstlane` ballot fold on AMD, and a shuffle-based
  emulation on Apple Silicon GPUs.

- Added `warp.match_all()`, which returns the warp's active-lane mask if every
  lane holds the same bits and 0 otherwise. It uses NVIDIA's `match.all.sync`
  instruction, a `readfirstlane` ballot fold on AMD, and a shuffle-based check
  on Apple Silicon GPUs.

- `DeviceGraphBuilder.collect_dependencies` now accepts an optional
  `dependencies` argument. The named predecessor handles are injected as
  ambient predecessors of every node the `work` closure adds, so the scope's
  nodes run after those predecessors without the closure threading the handles
  through to each `add_*` call. With the default (empty) `dependencies` the
  behavior is unchanged. When `work` adds no nodes, the returned join node
  falls back to depending on `dependencies` so it still chains correctly.

  ```mojo
  var producers = builder.collect_dependencies(add_producers)
  # Every node added by `add_consumers` depends on `producers`:
  var consumers = builder.collect_dependencies(
      add_consumers, dependencies=[producers]
  )
  ```

- Added a `DeviceGraphBuilder.add_function` overload that takes the kernel as a
  compile-time parameter and compiles it automatically, mirroring the
  parameter-based `DeviceContext.enqueue_function`. Callers no longer need a
  separate `DeviceContext.compile_function` step to add a kernel node:

  ```mojo
  def build(mut builder: DeviceGraphBuilder) raises {read}:
      _ = builder.add_function[kernel](
          42, grid_dim=1, block_dim=1, dependencies=[]
      )
  ```

- `AddressSpace` is now target-extensible rather than a fixed, portable enum.
  The built-in GPU spaces (`GENERIC`, `GLOBAL`, `SHARED`, `CONSTANT`, `LOCAL`,
  `SHARED_CLUSTER`, `BUFFER_RESOURCE`) are unchanged, but accessing any other
  name — for example an accelerator-specific `AddressSpace.SCRATCHPAD` — now
  resolves through the active hardware backend instead of being a hard-coded
  compile error. The set of valid address-space names is the union of the
  built-in GPU spaces and whatever the active backend defines, so accelerator
  backends can provide their own named spaces (with their own values) only
  where they exist. A name that no backend defines remains a compile-time
  error.

- Added support for the Steam Deck's RDNA2 Van Gogh APU.

## Removed

- Removed the `store_volatile()` and `load_volatile()` intrinsics from
  `std.gpu.intrinsics`. Use `UnsafePointer.store[volatile=True]()` and
  `UnsafePointer.load[volatile=True]()` instead, which work across all
  supported GPU targets rather than NVIDIA only.

- Removed the deprecated `GPUAddressSpace` alias for `AddressSpace`. Use
  `AddressSpace` directly.

## Fixed

- A `comptime` member with a trailing `where` clause is now accepted as a
  witness for a conditional trait conformance when the conformance constraint
  implies the member's constraint, for example:

  ```mojo
  trait StaticSize:
      comptime SIZE: Int

  struct Foo[size: Int = -1](StaticSize where size >= 0):
      comptime SIZE: Int where Self.size >= 0 = Self.size
  ```
