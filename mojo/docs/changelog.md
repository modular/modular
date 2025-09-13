# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### 🔥 Legendary

- Mojo now has support for default trait methods, allowing traits to provide
  reusable behavior without requiring every conforming struct to re-implement it.
  Default methods are automatically inherited by conforming structs unless
  explicitly overridden. For example:

  ```mojo
  # Any struct conforming to EqualityComparable now only needs to define one of
  # __ne__ ~or~ __eq__ and will get a definition of the other with no
  # additional code!

  # For instance:
  trait EqualityComparable:
      fn __eq__(self, other: Self) -> Bool:
          ...

      fn __ne__(self, other: Self) -> Bool:
          return not self == other

  @value
  struct Point(EqualityComparable):
      var x: Int
      var y: Int

      fn __eq__(self, other: Self) -> Bool:
          # Since __eq__ is implemented we now get __ne__ defined for free!
          return self.x == other.x and self.y == other.y

      # Defaulted methods can also be overriden if we want different behavior.
      # fn __ne__(self, other: Self) -> Bool:
      #     return self.x != other.x or self.y != other.y
  ```

  Currently a trait method is considered to be non-defaulted if the first thing in
  it's body is either a '...' or a 'pass' i.e.

  ```mojo

  trait Foo:
    # Either of the following are non-defaulted
    # fn foo(self):
    #   ...
    #
    # fn foo(self):
    #   pass

    # While this is not:
    fn foo(self):
      print("Foo.foo")
  ```

  Note that in the future only '...' will mark a trait method as not defaulted.

### Documentation

- New [Mojo vision](/mojo/vision) doc explains our motivations and design
decisions for the Mojo language.

- New [Mojo roadmap](/mojo/roadmap) provides a high-level roadmap for the
language across multiple phases.

### Language enhancements

- Methods on structs may now declare their `self` argument with a `deinit`
  argument convention.  This argument convention is used for methods like
  `__del__` and `__moveinit__` to indicate that they tear down the corresponding
  value without needing its destructor to be run again. Beyond these two
  methods, this convention can be used to declare "named" destructors, which are
  methods that consume and destroy the value without themselves running the
  values destructor.  For example, the standard `VariadicPack` type has these
  methods:

  ```mojo
  struct VariadicPack[...]:
      # implicit destructor
      fn __del__(deinit self): ...
      # move constructor
      fn __moveinit__(out self, deinit existing: Self): ...
      # custom explicit destructor that destroys "self" by transferring all of
      # the stored elements.
      fn consume_elements[
        elt_handler: fn (idx: Int, var elt: element_type) capturing
    ](deinit self): ...
  ```

  This argument convention is a fairly narrow power-user feature that is
  important to clarify the destruction model and make linear types fit into the
  model better.  A linear types are just types where all of the destructors are
  explicit - it has no `__del__`.

- Uncaught exceptions or segmentation faults in Mojo programs can now
  generate stack traces. This is currently only for CPU-based code. To generate
  a fully symbolicated stack trace, set the `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
  environment variable, use `mojo build` with debug info enabled, e.g.
  `-debug-level=line-tables`, and then run the resulting binary.

- Mojo now allows the use of keywords in function names (after `def` and `fn`)
  and in attribute references after a `.`. This notably allows the use of the
  `match` method in regex libraries even though Mojo takes this as a hard
  keyword.  Uses in other locations can still use backticks:

  ```mojo
  struct MatchExample:
      fn match(self): # This is ok now.
          pass

  fn test_match(a: MatchExample):
      a.match() # This is ok now.
      a.`match`() # This is still valid.
  ```

- When generating error messages for complex types involving parameter calls,
  the Mojo compiler now prints functions parameter values correctly, eliminating
  a large class of `T != T` errors that happen with GPU layouts.

### Language changes

- The `__del__` and `__moveinit__` methods should now take their `self` and
  `existing` arguments as `deinit` instead of either `owned`.

- The Mojo compiler now warns about use of the deprecated `owned` keyword,
  please move to `var` or `deinit` as the warning indicates.

- The `__disable_del` keyword and statement has been removed, use `deinit`
  methods instead.

- The previously deprecated `@value` decorator has been removed.

- Accesses to associated aliases and methods within a trait now require
  qualified references (prepended with `Self.`), making it consistent with how
  accesses to member aliases and methods in a struct require `self.`.

- The Mojo compiler now raises error on implicitly materialization of a
  non-`ImplicitlyCopyable` object, please either mark the type to be
  `ImplicitlyCopyable` or using `materialize[value: T]()` to explicitly
  materialize the parameter into a dynamic value.

### Standard library changes

- `Iterable`'s `origin` parameter is now named `iterable_origin`
  and its `mut` param is now named `iterator_mut` to avoid naming collisions.

- `zip` and `enumerate` are now builtins.

- Added `Path(...).parts()` method to the `Path` type, for example instead of
  writing:

  ```mojo
  var path = Path("path/to/file")
  var parts = path.path.split(DIR_SEPARATOR)
  ```

  you can now write:

  ```mojo
  var path = Path("path/to/file")
  var parts = path.parts()
  ```

- Added `Path(..).name()` method to the `Path` type, which returns the name of
  the file or directory.

- The `index()` free function now returns an `Int`, instead of a raw MLIR
  `__mlir_type.index` value.

- There is now an `iter` module which exposes the `next`, `iter`,
  `zip`, and `enumerate` methods.

- The way copying is modeled in Mojo has been overhauled.

  Previously, Mojo had two traits for modeling copyability:

  - `Copyable` denoted a type that could be copied implicitly
  - `ExplicitlyCopyable` denoted a type that could only be copied with an
    explicit call to a `.copy()` method.

  The vast majority of types defaulted to implementing `Copyable` (and therefore
  were implicitly copyable), and `ExplicitlyCopyable` was partially phased in
  but had significant usage limitations.

  Now, the new `Copyable` trait instead represents a type that can be
  *explicitly* copied (using `.copy()`), and a new `ImplicitlyCopyable` "marker"
  trait can be used to *opt-in* to making a type implicitly copyable as well.
  This swaps the default behavior from being implicitly copyable to being only
  explicitly copyable.

  The new `ImplicitlyCopyable` trait inherits from `Copyable`, and requires
  no additional methods. `ImplicitlyCopyable` is known specially to the
  compiler. (`ImplicitlyCopyable` types may also be copied explicitly using
  `.copy()`.)

  This makes it possible for non-implicitly-copyable types to be used with all
  standard library functionality, resolving a long-standing issue with Mojo
  effectively forcing implicit copyability upon all types.
  This will enable Mojo programs to be more efficient and readable, with fewer
  performance and correctness issues caused by accidental implicit copies.

  With this change, types that conform to `Copyable` are no longer implicitly
  copyable:

  ```mojo
  @fieldwise_init
  struct Person(Copyable):
      var name: String

  fn main():
      var p = Person("Connor")
      var p2 = p           # ERROR: not implicitly copyable
      var p3 = p.copy()    # OK: may be copied explicitly
  ```

  To enable a type to be implicitly copyable, declare a conformance to the
  `ImplicitlyCopyable` marker trait:

  ```mojo
  @fieldwise_init
  struct Point(ImplicitlyCopyable):
      var x: Float32
      var y: Float32

  fn main():
      var p = Point(5, 10)
      var p2 = p           # OK: may be implicitly copied
      var p3 = p.copy()    # OK: may be explicitly copied
  ```

  An additional nuance is that `ImplicitlyCopyable` may only be synthesized
  for types whose fields are all themselves `ImplicitlyCopyable` (and not
  merely `Copyable`). If you need to make a type with any non-`ImplicitlyCopyable`
  fields support implicit copying, you can declare the conformance to
  `ImplicitlyCopyable`, but write the `__copyinit__()` definition manually:

  ```mojo
  struct Container(ImplicitlyCopyable):
      var x: SomeCopyableType
      var y: SomeImplicitlyCopyableType

      fn __copyinit__(out self, existing: Self):
          self.x = existing.x.copy()   # Copy field explicitly
          self.y = existing.y
  ```

  - The following standard library types and functions now require only
    explicit `Copyable` for their element and argument types, enabling their use
    with types that are not implicitly copyable:
    `List`, `Span`, `InlineArray`, `Optional`, `Variant`, `Tuple`, `Dict`,
    `Set`, `Counter`, `LinkedList`, `Deque`, `reversed`.

    Additionally, the following traits now require explicit `Copyable` instead
    of `ImplicitlyCopyable`:
    `KeyElement`, `IntervalElement`, `ConvertibleFromPython`

  - The following Mojo standard library types are no longer implicitly copyable:
    `List`, `Dict`, `DictEntry`, `OwnedKwargsDict`, `Set`, `LinkedList`, `Node`
    `Counter`, `CountTuple`, `BitSet`, `UnsafeMaybeUninitialized`, `DLHandle`,
    `BenchConfig`, `BenchmarkInfo`, `Report`, `PythonTypeBuilder`.

    To create a copy of one of these types, call the `.copy()` method explicitly:

    ```mojo
    var l = List[Int](1, 2, 3)

    # ERROR: Implicit copying of `List` is no longer supported:
    # var l2 = l

    # Instead, perform an explicit copy:
    var l2 = l.copy()
    ```

    Alternatively, to transfer ownership,
    [use the `^` transfer sigil](https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-var-and-):

    ```moj
    var l = List[Int](1, 2, 3)
    var l2 = l^
    # `l` is no longer accessible.
    ```

  - User types that define a custom `.copy()` method must be updated to move
    that logic to `__copyinit__()`. The `.copy()` method is now provided by a
    default trait implementation on `Copyable` that should not be overridden:

    ```mojo
    trait Copyable:
        fn __copyinit__(out self, existing: Self, /):
            ...

        fn copy(self) -> Self:
            return Self.__copyinit__(self)
    ```

- A new `Some` utility is introduced to reduce the syntactic load of declaring
  function arguments of a type that implements a given trait or trait
  composition. For example, instead of writing

  ```mojo
  fn foo[T: Intable, //](x: T) -> Int:
      return x.__int__()
  ```

  one can now write:

  ```mojo
  fn foo(x: Some[Intable]) -> Int:
      return x.__int__()
  ```

- The comparison operators (e.g. `__eq__` and `__le__`) of the `SIMD` type now
  return a single `Bool` instead of a boolean `SIMD` mask. Moreover, `SIMD` now
  has explicit elementwise comparisons that return boolean masks, e.g. `eq()`
  and `le()`.
  - This allows `SIMD` to conform to the `EqualityComparable` trait, enabling
    the use of `SIMD` vectors in sets, as keys to dictionaries, generic search
    algorithms, etc. Moreover, `Scalar` now conforms to the `Comparable` trait,
    i.e. `SIMD` conforms to `Comparable` when the size is 1.
  - As a consequence, `SIMD.__bool__` no longer needs to be restricted to
    scalars, and instead performs an `any` reduction on the elements of vectors.

- Non-scalar `SIMD` constructors no longer allow implicit splatting of `Bool`
  values. This could lead to subtle bugs that cannot be caught at compile time,
  for example:

  ```mojo
  fn foo[w: Int](v: SIMD[_, w]) -> SIMD[DType.bool, w]:
    return v == 42  # this silently reduced to a single bool, and then splat
  ```

  Similarly to `InlineArray`, an explicit constructor with the `fill`
  keyword-only argument can be used to express the same logic more safely:

  ```mojo
  ```mojo
  fn foo[w: Int](v: SIMD[_, w]) -> SIMD[DType.bool, w]:
    return SIMD[DType.bool, w](fill=(v == 42))  # highlights the splat logic

  fn bar(Scalar[_]) -> Scalar[DType.bool]:
    # still works, since implicit splatting to a scalar is never ambiguous
    return v == 42
  ```

- Several types that wrap MLIR types have been changed to further
  encapsulate their behavior, hiding this low-level behavior from non-advanced
  users.

  - Types that can be constructed from raw MLIR values now require the use
    of an `mlir_value` keyword-only argument initializer.
    Affected types include: `SIMD`, `UInt`.

  - Types with raw MLIR type fields have had their `value` fields renamed to
    `_mlir_value`.
    Affected types include: `Bool`, `DType`.

- Added `os.path.realpath` to resolve symbolic links to an absolute path and

  remove relative path components (`.`, `..`, etc.). Behaves the same as the
  Python equivalent function.

- `Span` is now `Representable` if its elements implement trait
  `Representable`.

- `Optional` and `OptionalReg` can now be composed with `Bool` in
  expressions, both at comptime and runtime:

  ```mojo
  alias value = Optional[Int](42)

  @parameter
  if CompilationTarget.is_macos() and value:
      print("is macos and value is:", value.value())

- Added `sys.info.platform_map` for specifying types that can have different

  values depending on the platform:

  ```mojo
  from sys.info import platform_map

  alias EDEADLK = platform_map["EDEADLK", linux = 35, macos = 11]()
  ```

- Deprecated the following functions with `flatcase` names in `sys.info`:
  - `alignof`
  - `bitwidthof`
  - `simdbitwidth`
  - `simdbytewidth`
  - `simdwidthof`
  - `sizeof`

  in favor of `snake_case` counterparts, respectively:
  - `align_of`
  - `bit_width_of`
  - `simd_bit_width`
  - `simd_byte_width`
  - `simd_width_of`
  - `size_of`

- Added support for AMD RX 6900 XT consumer-grade GPU.

- Added support for AMD RDNA3.5 consumer-grade GPUs in the `gfx1150`,
`gfx1151`, and `gfx1152` architectures. Representative configurations have been
added for AMD Radeon 860M, 880M, and 8060S GPUs.

- Added support for NVIDIA GTX 1080 Ti consumer-grade GPUs.

- Added support for NVIDIA Tesla P100 datacenter GPUs.

- Updated `layout_tensor` copy related functions to support 2D and 3D
  threadblock dimensions.

- The `compile.reflection.get_type_name` utility now has limited capability to
  print parametric types, e.g. `SIMD[DType.float32, 4]` instead of just `SIMD`.
  If the parameter is not printable, an `<unprintable>` placeholder is printed
  instead. A new `qualified_builtins` flag also allows users to control the
  verbosity for the most common (but not all) builtin types.

- Add `repr` support for `List`, `Deque`, `Dict`, `LinkedList`, `Optional`, `Set`.
  [PR #5189](https://github.com/modular/modular/pull/5189) by rd4com.

- `InlineArray` now automatically detects whether its element types are
  trivially destructible to not invoke the destructors in its `__del__`
  function.  This improves performance for trivially destructible types
  (such as `Int` and friends).

- The `SIMD.from_bits` factory method is now a constructor, use
  `SIMD(from_bits=...)` instead.

- `String.splitlines()` now returns a `List[StringSlice]` instead of a
  `List[String]`. This avoids unnecessary intermediate allocations.

- `StringSlice.from_utf8` factory method is now a constructor, use
  `StringSlice(from_utf8=...)` instead.

- Added `os.atomic.fence` for creating atomic memory fences.
  ([#5216](https://github.com/modular/modular/pull/5216) by
  [@nate](https://github.com/NathanSWard))

  ```mojo
    from os.atomic import Atomic, Consistency, fence

    fn decrease_ref_count(ref_count: Atomic[DType.uint64]):
      if atomic.fetch_sub[ordering = Consistency.MONOTONIC](1) == 1:
        fence[Consistency.ACQUIRE]()
        # ...
  ```

- `Span` now implements a generic `.count()` method which can be passed a
  function that returns a boolean SIMD vector. The function counts how many
  times it returns `True` evaluating it in a vectorized manner. This works for
  any `Span[Scalar[D]]` e.g. `Span[Byte]`. PR
  [#3792](https://github.com/modularml/mojo/pull/3792) by [@martinvuyk](https://github.com/martinvuyk).

- Removed `alignment` and `static_alignment_cast` from `UnsafePointer`.

- Added `alignment` parameter to `UnsafePointer.alloc`.

### Tooling changes

- `mojo test` now ignores folders with a leading `.` in the name. This will
  exclude hidden folders on Unix systems ([#4686](https://github.com/modular/modular/issues/4686))

- `mojo doc --validate-doc-strings` now emits a warning when an `fn` function
is declared to raise an error (`raises`) and it has no [`Raises`
docstring](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/docstring-style-guide.md#errors).
However, because Mojo automatically treats all `def` functions as [raising
functions](/mojo/manual/functions#raising-and-non-raising-functions), we do not
enforce `Raises` docs for `def` functions (to avoid noisy false positives).

- Nightly `mojo` Python wheels are now available. To install everything needed
  for Mojo development in a Python virtual environment, you can use:

  ```sh
  pip install --pre mojo \
   --index-url https://dl.modular.com/public/nightly/python/simple/
  ```

  For more information, see the [Mojo install guide](/mojo/manual/install).

- In preparation for a future Mojo 1.0, the `mojo` and `mojo-compiler` packages
now have a `0.` prefixed to the version. Until the previous nightly packages
and 25.5 on Conda have been removed or yanked, we recommend specifying `<1.0.0`
as the version for these packages.

### Kernels changes

- A fast matmul for SM100 is available in Mojo. Please check it out in `matmul_sm100.mojo`.

- Moved `mojo/stdlib/stdlib/gpu/comm/` to `max/kernels/src/comm/`

### ❌ Removed

- The Mojo MLIR C bindings has been removed. This was a private package that was
 used for early experimentation.

### 🛠️ Fixed

- Fixed <https://github.com/modular/modular/issues/4695> - `Dict.__getitem__`
  always returns immutable references.
- Fixed <https://github.com/modular/modular/issues/4705> - Wrong mutability
  inferred for `__getitem__` if `[]` operator is used and `__setitem__` is present.
- Fixed <https://github.com/modular/modular/issues/5190>
- Fixed <https://github.com/modular/modular/issues/5139> - Crash on malformed initializer.
- Fixed <https://github.com/modular/modular/issues/5183> - Log1p not working on GPUs.
- Fixed <https://github.com/modular/modular/issues/5105> - Outdated `CLAUDE.md`
  docs.
- Fixed <https://github.com/modular/modular/issues/5239> - Contextual type not
  detected inside an inline if-else.
- Fixed <https://github.com/modular/modular/issues/5305> - Parser Segfaults on
  `LayoutTensor[layout]` with no `layout` in scope.
- Error messages involving types using implicit parameters from
  auto-parameterized types now include context information to solve a class of
  incorrect "T != T" error messages common in kernel code.
- Parameter inference failures now refer to parameters by their user-provided
  name, rather than complaining about a mysterious "parameter #4".
