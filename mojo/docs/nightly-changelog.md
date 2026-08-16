---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

- Code that performs many implicit conversions, most visibly large collection
  literals, compiles faster: the compiler no longer runs parameter inference on
  constructors that cannot be used for an implicit conversion in the first
  place. Files that are mostly data, such as the standard library's Unicode
  lookup tables, compile about 1.3x faster.

## Documentation

## Language enhancements

- A `thin` function type can now carry trailing `where` clauses, constraining
  the parameters it declares. This lets a generic algorithm state what it
  promises the function it is handed, instead of leaving the constraint to be
  restated at every binding site.

  ```mojo
  comptime Kernel = def[w: Int](Int) thin -> None where (
      w > 0, "width must be positive"
  )

  def apply[F: Kernel](x: Int):
      F[4](x)     # ok
      F[0](x)     # error: violated constraint
  ```

  The clause binds to the innermost function type, so a declaration-level
  `where` that follows a function-type result needs that result parenthesized:

  ```mojo
  def make[n: Int]() -> (def() thin -> None) where n > 0: ...
  ```

## Language changes

- Renamed the `@parameter` decorator on parametric closures to
  `@__parameter`. The deprecated `@parameter if` / `@parameter for`
  forms are unchanged; prefer `comptime if` / `comptime for` for
  compile-time control flow.

- The module & package system:

  - Directories may now have "namespace" semantics; a single directory name may
    resolve across distinct locations on disk which share that name.

    ```mojo
    # .
    # ├── one
    # │   └── foo
    # │       └── bar.mojo
    # └── two
    #     └── foo
    #         └── baz.mojo
    #
    # Compiles with -Ione -Itwo
    import foo.bar
    import foo.baz
    ```

  - Importing functions with the same name from different modules, combining
    them into one overload set, is now an error, following a period of
    deprecation.

  - Intra-package accesses without explicit `import`s are now an error,
    following a period of deprecation.

## Library stabilizations

## Library changes

- `ascii()` now pre-allocates capacity to avoid repeated reallocations when
  building the result string.
- Added `Dict.clear_with(destroy_func)`, the closure counterpart of `clear()`.
  Instead of destroying each entry in place, it hands the key and value to
  `destroy_func`, so it can clear a `Dict` whose key or value type is not
  `ImplicitlyDeletable`. The dictionary's capacity is retained, so it stays
  reusable.

- Added `Dict.insert(key, value)`, which stores a key/value pair and returns
  the displaced entry as an `Optional[DictEntry]` (empty when the key was not
  already present). Unlike `dict[key] = value`, `insert` does not destroy the
  displaced entry; it returns it, and the caller must destroy the returned
  entry. This is what lets `insert` work when the key or value type is not
  `ImplicitlyDeletable`:

  ```mojo
  var d = Dict[Int, Int]()
  var displaced = d.insert(1, 10)  # None — key 1 was absent
  displaced = d.insert(1, 20)      # the displaced (1, 10) entry
  ```

- By-reference `Dict` iteration (`for entry in dict`, `keys()`, `values()`,
  `items()`, and `reversed()`) no longer requires the key and value types to be
  `ImplicitlyDeletable`. These iterators only borrow references and never
  destroy an entry, so they now work on a `Dict` whose key or value type is not
  `ImplicitlyDeletable`. Consuming iteration (`for entry in dict^` and
  `take_items()`) still requires `ImplicitlyDeletable`, since it drops the
  entries it does not yield.

- The container backing variadic `**kwargs` has been renamed from
  `OwnedKwargsDict` to `StringDict`. `StringDict` no longer
  requires its value type `V` to be `ImplicitlyDeletable`. A keyword dictionary
  whose values are linear (non-`ImplicitlyDeletable`) is itself linear and must
  be torn down explicitly with the new `deinit_with(deinit_func)`, which hands
  each key and value to `deinit_func`. It also gained `insert(key, value)`
  (returns the displaced entry as an `Optional[DictEntry]` without destroying
  it) and `popitem()` (moves out and returns a whole entry), mirroring `Dict`.
  Operations that destroy a displaced value in place — `kwargs[key] = value` and
  the two-argument `pop(key, default)` — still require `V` to be
  `ImplicitlyDeletable`; use `insert`, `popitem`, or the single-argument
  `pop(key)` for linear values.

- `Coord` now conforms to `DevicePassable`, so a `Coord` embedded in a
  `DevicePassable` type (such as a `TileTensor`'s `Layout`) is encoded to the
  device through `Coord._to_device_type` instead of a raw field bit-copy, the
  same way `IndexList` already was.

- `reversed()` now works on typed ranges such as
  `reversed(range(Int16(1), 10, 2))`. The `ReversibleRange` trait gained an
  associated `ReversedType` iterator instead of hard-coding its `__reversed__()`
  return type, so every range flavor (including the typed scalar ranges) can
  conform and return its own reversed iterator.

- Added `to_numpy_array` and `from_numpy_array` to the new `python.numpy` module
  for moving flat numeric data between Mojo `Span`/`List` and NumPy arrays
  without hand-written `ctypes` plumbing:

  ```mojo
  from std.python.numpy import from_numpy_array, to_numpy_array

  var values: List[Float64] = [1.0, 2.0, 3.0]
  var array = to_numpy_array(values)                 # NumPy array (copies)
  var span = from_numpy_array[DType.float64](array)  # borrow array as a Span
  ```

  Both support the fixed-width numeric dtypes. `to_numpy_array` copies its
  input into a new, independent array; `from_numpy_array` borrows the array's
  buffer zero-copy.

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

- `chdir` has been added to the `std.os` module and an `fchdir` method has been
  added to `io.FileDescriptor`. These are wrappers for the corresponding POSIX
  functions.

- `TypeList.all_conforms_to()` is now implemented in terms of `conforms_to()`,
  which supports parameter-list operands like `Ts.values`. As a result,
  `all_conforms_to()` constraints preserve the same proof structure as direct
  `conforms_to(Ts.values, Trait)` constraints, so the compiler can use them in
  conditional conformance implication checks and type refinement.

  This means conditional conformances can rely on trait hierarchy relationships
  for an entire type parameter pack. Previously, a type that conditionally
  conformed to `JsonSerializable` would also need to repeat the inherited
  `Serializable` condition:

  ```mojo
  trait Serializable:
      pass

  trait JsonSerializable(Serializable):
      pass

  struct Packet[*Ts: Movable](
      Serializable where Ts.all_conforms_to[Serializable](),
      JsonSerializable where Ts.all_conforms_to[JsonSerializable](),
      Movable,
  ):
      pass
  ```

  Now the `JsonSerializable` condition is enough for the compiler to prove the
  inherited `Serializable` conformance:

  ```diff
   struct Packet[*Ts: Movable](
  -    Serializable where Ts.all_conforms_to[Serializable](),
       JsonSerializable where Ts.all_conforms_to[JsonSerializable](),
       Movable,
   ):
       pass
  ```

  The same constraints now refine each element of a variadic type parameter
  pack inside `where`, `comptime assert`, and `comptime if` contexts:

  ```mojo
  def write_all[*Ts: Movable](mut writer: Some[Writer], *args: *Ts):
      comptime if Ts.all_conforms_to[Writable]():
          comptime for i in range(args.__len__()):
              args[i].write_to(writer)
  ```

- `ImplicitlyDestructible` has been renamed to `ImplicitlyDeletable`, for better
  name consistency with its required `__del__()` "delete" special method.

- `is_trivially_destructible()` has been renamed to `is_trivially_deletable()`,
  for consistency with the `ImplicitlyDeletable` rename. It now also accepts any
  type (`T: AnyType`) instead of requiring `T: ImplicitlyDeletable`, returning
  `False` for non-`ImplicitlyDeletable` (linear) types.

- `List.resize` and `List.shrink` `new_size` arguments have been renamed to
  `new_length`.

- The `value` argument of `List.resize` has been renamed to `fill` to match
  List's constructor.

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
  collection^.deinit_with(my_destroy_closure)
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
  - `Dict[KeyType, ValueType, HasherType]`
    - Element-destroying and key/value-copying operations (`__setitem__`,
      `setdefault`, `fromkeys`, `update`, `__or__`, `__ior__`, `pop`, `clear`)
      still require the `K` key and `V` value types to be `ImplicitlyDeletable`,
      so a `Dict` with non-`ImplicitlyDeletable` keys or values can currently be
      constructed and torn down with `destroy_with()` but not populated or
      mutated. For deletable key/value types (the common case) this is
      transparent.
    - Consuming iteration (`for entry in dict^`) is likewise conditional,
      requiring `ValueType` to be `ImplicitlyDeletable`.
  - `LinkedList[ElementType]`
    - Unlike `Dict`, a `LinkedList` with non-`ImplicitlyDeletable` elements can
      be populated (`append`, `prepend`, `insert`, `extend`) and then torn down
      with `destroy_with()`.
    - Only `clear` still requires `ElementType` to be `ImplicitlyDeletable`. For
      deletable element types (the common case) this is transparent.
    - `LinkedList.insert()` no longer raises on an out-of-range index; like
      `List.insert()`, it now aborts (checked when asserts are enabled).
    - Consuming iteration (`for x in list^`, the `IterableOwned` conformance)
      is likewise conditional, requiring `ElementType` to be
      `ImplicitlyDeletable`.

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

- `coord` is now a comptime expression, and `coord[DType]()` has been renamed
  to `dyn_coord[DType]()`.
  Now one can just write:

   ```mojo
   var my_coord = coord[1, 2, 3]
   ```

   to create a `Coord[ComptimeInt[1], ComptimeInt[2], ComptimeInt[3]]`

- Removed `trait_downcast_var()`. Improvements to type refinement based on
  `where conforms_to(..)` and `comptime assert conforms_to(..)` make explicit
  value trait downcasting no longer necessary.

- The `ConditionalType` type function in `std.utils.type_functions` is now
  deprecated. Use the equivalent ternary expression `T if cond else U`
  instead:

  ```mojo
  # Deprecated:
  comptime Storage = ConditionalType[If=cond, Then=Int, Else=NoneType]

  # Use instead:
  comptime Storage = Int if cond else NoneType
  ```

- Added `raise_python_exception()` to `std.python.bindings`, which translates a
  Mojo `Error` into a Python exception via `PyErr_SetString` and returns a null
  `PyObjectPtr`.

- Iterating over a `String`, `StringSlice`, or `StringLiteral` now yields
  grapheme clusters by default. Their `__iter__()` and `__reversed__()` methods
  return a `GraphemeSliceIter`, so `for c in my_string:` produces what a user
  perceives as a single "character" on screen. The lower-level views remain
  available when you want them: `codepoints()` or `codepoint_slices()` for
  Unicode scalars, and `bytes()` for raw UTF-8 bytes.

- The `Equatable` trait now allows for positional-only implementations, and
  argument on implementers no longer need to match the trait exactly.

- `UnsafePointer.init_pointee_move()` and `UnsafePointer.init_pointee_copy()`
  are now deprecated in favor of a single `unsafe_write()` method. Moving a
  value in works the same as before:

  ```mojo
  ptr.unsafe_write(value^)
  ```

  To copy a value in instead of moving it, pass it as the `copy` keyword
  argument:

  ```mojo
  ptr.unsafe_write(copy=value)
  ```

- `UnsafePointer.destroy_pointee()` and `UnsafePointer.destroy_pointee_with()`
  are now deprecated in favor of the new `unsafe_deinit_pointee()` method, which
  covers both cases: call it with no arguments to destroy an
  `ImplicitlyDeletable` pointee, or pass a deinitializing closure to destroy a
  non-`ImplicitlyDeletable` pointee in place.

## Tooling changes

- Added a `--lld-path` CLI flag. This overrides the LLD path that Mojo uses.

- `mojo-lsp-server` no longer parses or type-checks code blocks inside
  docstrings by default. This checking rests on unstable foundations in the
  LSP server and was prone to failing, producing false-positive diagnostics
  unrelated to the code being edited, for little value in return. Pass
  `-check-docstrings` when launching `mojo-lsp-server` from the command line
  to re-enable the previous behavior. We plan to make this checking more
  robust and re-enable it by default over time.
- `StringDict` now conforms to `Writable` when its value type is `Writable`,
  matching the existing behavior of `Dict`. This lets you `print()` a
  `StringDict` or convert it to a `String`.

- The `chars` argument of `strip()`, `lstrip()` and `rstrip()` on `StringSpan`,
  `String` and `StringLiteral` is now an `ImmStringSpan`, so a mutable string
  is accepted as `chars`, including the string being stripped (`s.strip(s)`).

- `StringDict.__getitem__()` now accepts a `StringSpan`, so you can index a
  `StringDict` with a borrowed string view without first allocating a
  `String` just to perform the lookup.

- Renamed the variadic type-list parameter on `Tuple` and `VariadicPack` to
  `Ts`, standardizing the naming convention used across the standard library.
  The old name, `element_types`, remains as a deprecated alias.

- Added experimental `DType.float6_e2m3fn` and `DType.float6_e3m2fn`, the two
  6-bit encodings from the
  [Open Compute microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
  Both are finite-only, so neither has an inf nor a NaN encoding.

  These are experimental storage formats for packed weights rather than
  general-purpose numeric types, and standard library support is deliberately
  partial. As with the existing `DType.float4_e2m1fn`, they are excluded from
  `is_numeric()`, arithmetic is not implemented, and converting to or from
  another floating-point type is unsupported on every target, so values cannot
  be printed either.

- `Array` now conforms to `Defaultable` when its type `T` is also `Defaultable`.

- Deprecated `is_trivially_movable()`, `is_trivially_copyable()`, and
  `is_trivially_deletable()` in `std.memory` in favor of
  `IsTriviallyMovable[T]`, `IsTriviallyCopyable[T]`, and
  `IsTriviallyDeinitable[T]` in `std.traits`. The replacements are `comptime`
  predicates rather than functions, so drop the call parens at use sites, for
  example `IsTriviallyCopyable[T]` instead of `is_trivially_copyable[T]()`.

- Renamed `UnsafeMaybeUninit` to `MaybeUninit`. It conforms to `Movable`,
  `Copyable`/`ImplicitlyCopyable`, and `Deinitable` only when the contained
  type's own move, copy, or implicit deinitializer is trivial, since
  moving, copying, or destroying a `MaybeUninit` only touches its raw bits,
  never the contained value's own lifecycle methods. Gating conformance this way
  turns what would otherwise be silent memory-safety bugs into compile-time
  errors.

- `Atomic` is now parameterized on a value type `T` instead of a `DType`.
  Update call sites from `Atomic[DType.float32]` to `Atomic[Float32]`. The
  atomic operations (`load()`, `store()`, `fetch_add()`, `compare_exchange()`,
  and so on) still only support `Scalar` types.

- Added `Pointer[T].unsafe_write(def() -> T)`, which initializes the pointee
  with the value returned by a closure, constructing it directly in place rather
  than moving an already-constructed value there. Unlike `unsafe_write(var T)`,
  this does not require the pointee type to be `Movable`.

- Added `write()` to `MaybeUninit` and `Pointer`, as a safe counterpart to
  `unsafe_write()` for types that are trivially deinitializable (for example
  `Int`). Since a trivial deinitializer is a no-op, overwriting a live value
  through `write()` can't leak a resource, so it's callable without first
  destroying the previous value. Prefer it over `unsafe_write()` whenever the
  pointee type is trivially deinitializable.

- `Pointer.mut_cast` is now deprecated. Developers should prefer using explicit
  mutabilites at the callsite via `MutPointer` or `ImmPointer`. If mut casting
  is needed (it should try to be avoided) - you can use `unsafe_mut_cast`.

- The following APIs have been migrated to unified closures: `sort`,
  `debug_assert`, `Span.apply`.

- Uncaught exceptions now print to `stderr`, not `stdout`.

## GPU programming

## Tooling changes

- `mojo doc` now reports the condition of a conditional trait conformance, and
  the generated API docs show it alongside the trait. Previously the condition
  was dropped, making a conditional conformance indistinguishable from an
  unconditional one. Also fixed rendering of some `where` clauses.

## Removed

This release completes the removal of APIs deprecated during the v1.0 cycle.

- Removed the temporary `InlineArray` alias for `Array`, including its
  re-exports from `std.collections` and the prelude. Use `Array` directly.

- Removed the `std.gpu.profiler` module and its `ProfileBlock` context manager.
  It timed host wall-clock, not GPU work, and reported the elapsed time with the
  operands reversed. Time a block of host code with
  [`perf_counter_ns()`](/docs/std/time/time/perf_counter_ns/) directly, and use
  a GPU profiler such as Nsight Systems or `rocprof` for device timings.

- Removed `memcmp` and its `std.memory` re-export. Use `unsafe_memcmp`
  instead.

- Removed `String.set_byte_length()`, an internal helper that set the length
  field without reserving capacity.

- Removed the `validate` parameter from
  [`b64decode()`](/docs/std/base64/base64/b64decode/), which now always
  validates. Passing `validate=False` did not skip any work on valid input; it
  only turned characters outside the base64 alphabet into silently corrupt
  output bytes. Drop `[validate=True]` from existing calls; calls that relied on
  the default now raise instead of returning garbage.

- Removed the origin aliases left over from the `Immut` to `Imm` and
  `External` to `Untracked` renames. Use the surviving spelling in each case:
  `ImmOrigin` for `ImmutOrigin`, `ImmUnsafeAnyOrigin` for
  `ImmutUnsafeAnyOrigin`, `ImmStaticOrigin` for `StaticConstantOrigin`,
  `UntrackedOrigin` for `ExternalOrigin`, `MutUntrackedOrigin` for
  `MutExternalOrigin`, and `ImmUntrackedOrigin` for both
  `ImmutUntrackedOrigin` and `ImmutExternalOrigin`.

- Removed the pre-unification pointer aliases `MutUnsafePointer`,
  `ImmUnsafePointer`, `ImmutUnsafePointer`, `ImmutOpaquePointer`,
  `ImmutPointer`, and `OptionalUnsafePointer`. Use `MutPointer`, `ImmPointer`,
  `ImmOpaquePointer`, and `OptionalPointer` instead. `UnsafePointer` itself
  remains available, but is deprecated in favor of `Pointer`.

- Removed the raw memory functions superseded by their `unsafe_`-prefixed
  spellings: `memcpy`, `memset`, `memset_zero`, `uninit_move_n`,
  `uninit_copy_n`, and `destroy_n`. Use `unsafe_memcpy`, `unsafe_memset`,
  `unsafe_memset_zero`, `unsafe_uninit_move_n`, `unsafe_uninit_copy_n`, and
  `unsafe_destroy_n` instead.

- Removed the `size` aliases left from the `size` to `length` rename:
  `SIMD.size`, `Array.size`, `TypeList.size`, and the `SIMDSize` alias for
  `SIMDLength`. Use `length` and `SIMDLength`.

- Removed the `as_immutable()` and `get_immutable()` methods on `Pointer`,
  `Span`, and `StringSpan`. Use `as_imm()`.

- Removed the `ImmutSpan` alias. Use `ImmSpan`.

- Removed `String.as_string_slice()`. Construct a `StringSpan` from the string
  instead: `StringSpan(my_string)`.

- Removed the `ImplicitlyDestructible` and `ImplicitlyDeletable` aliases. Use
  `Deinitable`.

- Removed the deprecated ownership-transfer methods: `List.steal_data()` and
  `OwnedPointer.steal_data()` are now `unsafe_take_allocation()`,
  `OwnedPointer.take()` is `into_inner()`, and `Variant.take()` and
  `Variant.unsafe_take()` are `unwrap()` and `unsafe_unwrap()`.

- Removed the `Pointer` methods superseded by their `unsafe_`-prefixed
  spellings: `as_noalias_ptr()`, `destroy_pointee()`, `destroy_pointee_with()`,
  `init_pointee_move()`, `init_pointee_copy()`, and `init_pointee_move_from()`.
  Use `unsafe_as_noalias()`, `unsafe_deinit_pointee()`,
  `unsafe_deinit_pointee_with()`, `unsafe_write()`, and
  `unsafe_write_move_from()`. The `Pointer.type` alias for `Pointer.T` is gone
  as well.

- Removed the `ConditionalType` type function and the `std.utils.type_functions`
  module. Use the ternary expression `T if cond else U`.

- Removed `trait_downcast()`. Constrain on the trait instead, with
  `conforms_to(type_of(src), Trait)` in a `where` clause or a
  `comptime assert`.

- Removed the parametric `benchmark.run[func]()` overloads. Pass the function as
  an argument to `run(f)` instead, which accepts a unified closure.

- Removed `AnyCoroutine`, `Coroutine` and `RaisingCoroutine` from the prelude,
  and made the module that defines them private. Mojo's async support is
  unfinished, and these types being globally visible led people to build on an
  API that carries no stability guarantees. `async def` is unaffected: the
  compiler still synthesizes these types for you, so they continue to appear in
  inferred types and diagnostics. There is no supported way to name them
  directly.

- Removed the async task API from the public `std.runtime.asyncrt` module,
  which is now private. `initialize_runtime()` and `parallelism_level()` are
  unaffected and have moved up to the `std.runtime` package, so import them
  from `std.runtime` instead of `std.runtime.asyncrt`.

- Removed support for `.mojopkg` files after a period of deprecation. Use
  `.mojoc` files instead.

## Fixed

- `mojo build --emit asm` and `--emit llvm` now always write the offload kernel
  files next to the host output file. Building a kernel that an earlier build
  had already compiled could write them into the earlier build's output
  directory, or skip them with no diagnostic.

- Parametric `raises` now accepts any primary expression as the thrown type in
  a function signature, matching the syntax positions where types otherwise
  appear. This most notably fixes `raises Self.SomeAssocType` on trait and
  struct methods, which would previously fail with an error. The parenthesized
  workaround (`raises (Self.DriveErrorType)`) is no longer required.

- An integer `range()` with a step of zero is now always empty. It previously
  used to be an infinite loop - iterating forever at runtime, and hanging the
  compiler at comptime.

- Fixed `ceildiv()` returning `0` for unsigned operands near the type's
  maximum value. The unsigned code path computed `numerator + denominator -
  1`, which overflows and wraps for large operands; it now derives the
  ceiling from the floor division and remainder instead.

- `Counter.most_common(n)` now returns all elements when `n` exceeds the
  number of unique elements, matching Python, instead of aborting.

- `os.path.join()` now inserts separators based on the accumulated path rather
  than the first argument, so `join("/", "a", "b")` returns `/a/b` (previously
  `/ab`) and `join("a", "b/", "c")` returns `a/b/c` (previously `a/b//c`).

- `base64.b64decode()` now raises an error when the input length is not
  divisible by 4 instead of reading past the end of the input (or aborting
  when asserts are enabled).

- On macOS, `os.stat()` and `os.lstat()` no longer return a negative
  `st_mode` for regular files. The underlying `mode_t` and `nlink_t` C type
  aliases were declared as signed 16-bit integers, but macOS defines them as
  unsigned, so any mode with the `S_IFREG` bit set (every regular file)
  sign-extended into a negative `Int`.

- `PythonObject` no longer leaks a CPython reference per positional argument
  when calling a Python object, nor when setting an item, attribute, or set
  literal element.
