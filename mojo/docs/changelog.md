# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements

- Mojo now supports raising "typed errors", where a function can specify a
  what type it raises instead of defaulting to the `Error` type. This is done by
  specifying it after the `raises` keyword, e.g.
  `fn foo() raises CustomError -> Int`.

  Raised errors in Mojo are very efficient - they work as an alternate return
  value: for example, a function like `fn () raises Int -> Float32:` compiles
  into code that returns either an `Int` or a `Float32` and uses an implicit
  boolean result to determine which one is valid - there is no expensive stack
  unwinding or slow dynamic logic that is implied.  This means that thrown
  errors work fine on GPUs and other embedded targets.

  The 'caught' type in a `try` block is automatically inferred to be the first
  thrown type inside of the `try` body, e.g.:

  ```mojo
  try:
      print(foo())
  except err:       # "err" is typed as CustomError
      print(err)
  ```

  Typed throws "just work" with generics, allowing the definition of higher
  order functions like:

  ```mojo
  fn parametric_raise_example[ErrorType: AnyType](fp: fn () raises ErrorType) raises ErrorType:
      # ... presumably some iteration or other exciting stuff happening here.
      fp()
  ```

  This dovetails with other support to allow contextually generic thrown types,
  e.g.:

  ```mojo
  fn call_parametric_raise_example[GenTy: AnyType](func_ptr: fn () raises GenTy):
    fn raise_int() raises Int: pass
    try:
        parametric_raise_example(raise_int)
    except err_int:   # Typed as Int
        ref x: Int = err_int

    fn raise_string() raises String: pass
    try:
      parametric_raise_example(raise_string)
    except err_string: # Typed as String
        ref s: String = err_string

    try:
      parametric_raise_example(func_ptr)
    except err_gen: # Typed as GenTy
        ref s: GenTy = err_gen

    # Non-raising functions infer an error type of `Never`, allowing these
    # functions to propagate non-raisability across generic higher-order
    # functions conveniently.
    fn doesnt_raise(): pass
    # Note this isn't in a try block. Mojo knows 'parametric_raise_example'
    # doesn't raise because the 'doesnt_raise' function doesn't.
    parametric_raise_example(doesnt_raise)
  ```

  This support should be reliable, but `with` blocks are still hard coded to
  `Error`.

- Mojo now allows implicit conversions between function types from a non-raising
  function to a raising function.  It also allows implicit conversions between
  function types whose result types are implicitly convertible.

  ```mojo
  fn takes_raising_float(a: fn () raises -> Float32): ...
  fn returns_int() -> Int: ...
  fn example():
      # This is now ok.
      takes_raising_float(returns_int)
  ```

- Mojo now differentiates between `...` and `pass` in trait methods. The use of

  `...` continues to denote no default implementation - `pass` now specifies a
  default do-nothing implementation. For example:

  ```mojo
  trait T:
      # No default implementation
      fn foo(self): ...

      # Default implementation that does nothing
      fn bar(self) : pass
  ```

  The compiler will error on the use of `pass` to define a default
  implementation for a trait method with results:

  ```mojo
  trait T:
      foo.mojo:2:26: error: trait method has results but default implementation returns no value; did you mean '...'?
      fn foo(self) -> Int: pass
                           ^
      trait.mojo:2:8: note: in 'foo', declared here
      fn foo(self) -> Int: pass
         ^
  ```

- Mojo now supports a `Never` type, which can never be instantiated.
  This type can be used for functions (like `abort()`) which do not have a
  normal return value, and for functions that are guaranteed to raise without
  returning a normal value.  Functions that are declared to raise `Never` (and
  generic functions instantiated with `Never` as their error type) compile into
  the same ABI as functions that don't `raise`.

- The `deinit` argument convention can now be applied to any argument of a
  struct method, but the argument type still must be of the enclosing struct
  type.

### Language changes

- The compiler will now warn on unqualified access to struct parameters, e.g.

  ```mojo
  @fieldwise_init
  struct MyStuff[my_param: Int]:
      fn give_me_stuff(self) -> Int:
          # Warning: unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead
          return my_param
  ```

### Library changes

- The `Copyable` trait now refines the `Movable` trait.  This means that structs
  and generic algorithms that already require `Copyable` don't need to also
  mention they require `Movable.

- `Span` now conforms to `Iterable`.

- `any` and `all` now work over `Iterable`s,
  which means they can act over the result of `map`.

- Tuples have been improved:
  - Tuples can now be concatenated with `Tuple.concat(other)`.
  - Tuple can now be reversed with `Tuple.reverse()`.

- New `ContiguousSlice` and `StridedSlice` types were added to
  the `builtin_slice` module to support specialization for slicing without strides.

- `List` slicing without a stride now returns a `Span`, instead of a `List` and
  no longer allocates memory.

- Remove `List` variadic initializer.

  - Statements like:

    ```mojo
    var x = List[Int32](1, 2, 3)
    ```

    can be updated to:

    ```mojo
    var x: List[Int32] = [1, 2, 3]
    ```

  - Expressions like:

    ```mojo
    var x = foo(List[Float32](1, 2, 3))
    ```

    can be updated to move the explicit type "hint" around the first elememnt:

    ```mojo
      var x = foo([Float32(1), 2, 3])
    ```

  - Expressions like:

    ```mojo
    var data = Span(List[Byte](1, 2, 3))
    ```

    can be updated to move the explicit element type to the `Span`:

    ```mojo
    var data = Span[Byte]([1, 2, 3])
    ```

- The `random` module now uses a pure Mojo implementation based on the Philox
  algorithm (via an internal wrapper), replacing the previous `CompilerRT` C++
  dependency. The Philox algorithm provides excellent statistical quality, works
  on both CPU and GPU, and makes random number generation fully transparent and
  source-available. Note that this changes the random number sequence for a given
  seed value, which may affect tests or code relying on reproducible sequences.

- Implicit conversion between `Int` and `UInt` have been removed.

- `UnsafePointer` can now be initialized from a raw memory address using the
  `unsafe_from_address` initializer.

- `alloc()` now has a `debug_assert` ensuring count is non-negative.

- The `EqualityComparable` trait has been deprecated in favor of `Equatable`,
  which has identical functionality.

- `DLHandle.get_symbol()` and `OwnedDLHandle.get_symbol()` now return
  `UnsafePointer[T, MutAnyOrigin]` instead of `UnsafePointer[T, ImmutAnyOrigin]`.
  The vast majority of symbols loaded from shared libraries are meant to be used
  mutably, and it's safer to go from mutable → immutable (via `.as_immutable()`)
  than from immutable → mutable (via `.unsafe_mut_cast[True]()`). Users who need
  immutable pointers can now simply call `.as_immutable()` on the result.

- The `os` module now exposes a `link` function, wrapping the unix
  `link(2)` system call

- Added a `CStringSlice` as a type-safe way to interact with nul-terminated
  c-style strings (`const char*`).

- The `os` module now exposes a `symlink` function, wrapping the unix
  `symlink(2)` syscall.

- Various OS wraper functions now include the value of `errno` in the raised error
  message.

- The `ImplicitlyBoolable` trait has been removed. This trait enabled types to
  implicitly convert to `Bool`. This behavior was rarely used, and could lead
  to subtle bugs, for example mistakenly passing types like `Int` or
  `UnsafePointer` to an argument expecting a `Bool` would silently compile
  successfully.

- Using a new 'unconditional conformances' technique leveraging `conforms_to()`
  and `trait_downcast()` to perform "late" element type conformance checking,
  some standard library types are now able to conform to traits that they could
  not previously. This includes:

  - `List` now conforms to `Equatable`, `Writable`, `Stringable`,
    and `Representable`.

  - The following types no longer require their elements to be `Copyable`.
    - `Tuple`
    - `Variant`

- Basic file I/O operations in the `io` module are now implemented natively in
  Mojo using direct `libc` system calls (`open`, `close`, `read`, `write`,
  `lseek`). The `FileHandle` type no longer depends on CompilerRT functions,
  providing better performance and transparency. Error handling now includes
  errno-based messages for improved diagnostics.

- Removed `String.join(*Writable)` overload that takes a variadic sequence
  of arguments, as it could be ambiguous with the remaining
  `String.join(Span[Writable])` overload.

- Remove the `Int.__init__(self, value: StringSlice, base: UInt)` constructor.
  Users should call `atol` directly.

- `DeviceContext.enqueue_function_checked()` and
  `DeviceContext.enqueue_function_experimental()` now automatically infer
  `func_attribute` to `FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(shared_mem_bytes)`
  when `shared_mem_bytes` is specified but `func_attribute` is not, for NVIDIA GPUs
  with allocations > 48KB. This eliminates the need to specify the same shared memory
  size twice in many cases, reducing boilerplate and preventing mismatched values.
  On AMD GPUs or for allocations ≤ 48KB, explicit `func_attribute` values
  should be provided when needed.

### Tooling changes

- The Mojo compiler now "diffs" very long types in error messages to explain
  what is going on in a more easy to understand way.
- Specifying CUDA architectures with `--target-accelerator` now expects a sm
  version string rather than just a compute capability. For example
  `--target-accelerator=nvidia:80` should be changed to
  `--target-accelerator=nvidia:sm_80`. If an incorrect format is used for the
  version, the compiler will default to the lowest supported sm version.
- Elaboration error printing with different level of verbosity
  which offers control on how parameter values are displayed as part of
  elaboration errors when function instantiation fails.
  `--elaboration-error-verbose=value` now takes a value, where:
  - `no-params` means don't display any concrete parameter values.
    This is helpful to collapse recursion related error message
    into shorter blobs.
  - `simple-params` display concretized parameter values for simple types,
    including numeric types and strings, in a user-friendly format
    (default value).
  - `all-params` means show all concrete parameter values.
    This is for advanced programmer who doesn't mind reading
    MLIR attributes but wants more visibility of parameter values.
- `--elaboration-max-depth` is added to control maximum elaborator
   instantiation depth. This (unsigned) value helps to detect compile time
   recursion. The default is `std::numeric_limits<unsigned>::max()`.
- The Mojo Debugger `mojo break-on-raise` feature now works correctly with
  multiple targets in a debugger instance. The setting is per-target.

### Experimental changes

Changes described in this section are experimental and may be changed, replaced,
or removed in future releases.

- Mojo now supports compile-time trait conformance check (via `conforms_to()`)
  and downcast (via `trait_downcast()`). This allows users to implement features
  like static dispatching based on trait conformance. For example:

  ```mojo
  fn maybe_print[T : AnyType](maybe_printable : T):
    @parameter
    if conforms_to(T, Writable):
      print(trait_downcast[Writable](maybe_printable))
    else:
      print("[UNPRINTABLE]")
  ```

- Added support for `DType` expressions in `where` clauses:

  ```mojo
  fn foo[dt: DType]() -> Int where dt is DType.int32:
      return 42
  ```

  Currently, the following expressions are supported:
  - equality and inequality
  - `is_signed()`, `is_unsigned()`, `is_numeric()`, `is_integral()`,
    `is_floating_point()`, `is_float8()`, `is_half_float()`

- Added support for `SIMD` expressions in `where` clauses:

  ```mojo
  fn foo[dt: DType, x: Int]() -> Int where SIMD[dt, 4](x) + 2 > SIMD[dt, 4](0):
      return 42
  ```

  Currently, the following expressions are supported:
  - default construction and construction from `Int` and `IntLiteral`
  - equality, inequality, and other comparison operators
  - addition, subtraction, and multiplication
  - bitwise logical operations, excluding shifts

### ❌ Removed

### 🛠️ Fixed

- [Issue #5578](https://github.com/modular/modular/issues/5578): ownership
  overloading not working when used with `ref`.
- [Issue #5137](https://github.com/modular/modular/issues/5137): Tail call
  optimization doesn't happen for tail recursive functions with raises
