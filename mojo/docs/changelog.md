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

- Mojo now supports "typed throws", where a function can specify a specific
  error type that it raises instead of defaulting to the `Error` type by
  specifying it after the `raises` keyword in parentheses, e.g.
  `fn foo() raises (CustomError) -> Int`.

  Raised errors in Mojo are very efficient - they work as an alternate return
  value: for example, a function like `fn () raises (Int) -> Float32:` compiles
  into code that returns either an `Int` or a `Float32` and uses an implicit
  boolean result to determine which one is valid - there is no expensive stack
  unwinding or slow dynamic logic that is implied.  This means that thrown
  errors work fine on GPUs and other embedded targets.

  The 'caught' type in a `try` block is automatically inferred to be the first
  thrown type inside of the try body, e.g.:

  ```mojo
  try:
      print(foo())
  except err:       # "err" is typed as CustomError
      print(err)
  ```

  Typed throws "just work" with generics, allowing the definition of higher
  order functions like:

  ```mojo
  fn parametric_raise_example[ErrorType: AnyType](fp: fn () raises (ErrorType)) raises (ErrorType):
      # ... presumably some iteration or other exciting stuff happening here.
      fp()
  ```

  This dovetails with other support to allow contextually generic thrown types,
  e.g.:

  ```mojo
  fn call_parametric_raise_example[GenTy: AnyType](func_ptr: fn () raises (GenTy)):
    fn raise_int() raises (Int): pass
    try:
        parametric_raise_example(raise_int)
    except err_int:   # Typed as Int
        ref x: Int = err_int

    fn raise_string() raises (String): pass
    try:
      parametric_raise_example(raise_string)
    except err_string: # Typed as String
        ref s: String = err_string

    try:
      parametric_raise_example(func_ptr)
    except err_gen: # Typed as GenTy
        ref s: GenTy = err_gen
  ```

  This support should be reliable, but there are a few limitations: 1) `with`
  blocks are still hard coded to `Error`.  2) Thrown errors must exactly match
  the contextual thrown type, no implicit conversions are allowed. 3)
  Parentheses are required around the thrown type for now. 4) Mojo has no
  equivalent of the Swift `Never` type for making parametricly-raising functions
  be treated as non-throwing when working with non-throwing higher order
  functions.

- Mojo now allows implicit conversions between function types from a non-raising
  function to a raising function.  It also allows implicit conversions between
  function types whose result types are implicitly convertible:

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

- Added a `CStringSlice` as a type-safe way to interact with nul-terminated
  c-style strings (`const char*`).

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

### ❌ Removed

### 🛠️ Fixed

- [Issue #5578](https://github.com/modular/modular/issues/5578): ownership
  overloading not working when used with `ref`.
