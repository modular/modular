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

  As part of this, context managers have been extended to support typed throws,
  and can also infer an error type if they need to handle it, e.g.:

  ```mojo
  struct MyGenericExitCtxtMgr:
    # Called on entry to the with block.
    fn __enter__(self): ...
    # Called on exit from the with block when no error is thrown.
    fn __exit__(self): ...
    # Called on exit from the with block if an error is thrown.
    fn __exit__[ErrType: AnyType](self, err: ErrType) -> Bool: ...
  ```

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

- Mojo now allows the use of a `comptime(x)` expression to force a subexpression
  to be evaluated at compile time.  This can help make working with certain
  types more elegant when you can't (or don't want to) materialize them into a
  runtime value.  For example, if you just want the size from a compile time
  layout:

  ```mojo
  fn takes_layout[a: Layout]():
    # materializes entire layout value just to get the size out of it
    print(a.size())
    # Could already work around this with a comptime declaration, verbosely.
    comptime a_size = a.size()
    print(a_size)
    # Can now tell Mojo to evaluate the expression at comptime.
    print(comptime(a.size()))
  ```

- The `deinit` argument convention can now be applied to any argument of a
  struct method, but the argument type still must be of the enclosing struct
  type.

- Context managers (used in `with` statements) can now define consuming exit
  methods, i.e. `fn __exit__(var self)` which can be useful for linear context
  managers. This also works with `deinit`.

- Mojo now allows functions that return references to convert to functions that
  return values if the type is implicitly copyable or implicitly convertible to
  the destination type:

  ```mojo
  fn fn_returns_ref(x: SomeType) -> ref [x.field] Int: ...
  fn examples():
      # OK, Int result from fn_returns_ref can be implicitly copied.
      var f1 : fn (x: SomeType) -> Int = fn_returns_ref
      # OK, Int result from fn_returns_ref implicitly converts to Float64.
      var f2 : fn (x: SomeType) -> Float64 = fn_returns_ref
  ```

- Mojo now supports the `...` expression.  It is a logically empty value of
  `EllipsisType`.  It can be used in overloaded functions (e.g. getitem calls),
  e.g.:

  ```mojo
  struct YourType:
    fn __getitem__(self, idx: Int) -> Int:
      # ... behavior when passed x[i]
    fn __getitem__(self, idx: EllipsisType) -> Int:
      # ... behavior when passed x[...]
  ```

### Language changes

- The `*_` and `**_` syntax for explicitly unpacked parameters has been replaced
  with a simplified `...` syntax.  Instead of `T[4, 5, *_, **_]` you can now use
  `T[4, 5, ...]`.  The `...` delays binding of both keyword and non-keyword
  parameters.

- The compiler will now warn on unqualified access to struct parameters, e.g.

  ```mojo
  @fieldwise_init
  struct MyStuff[my_param: Int]:
      fn give_me_stuff(self) -> Int:
          # Warning: unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead
          return my_param
  ```

- The compiler will now warn on the use of `alias` keyword and suggest
`comptime` instead.

- The Mojo language basic trait hierarchy has changed to expand first-class
  support for linear types (aka. non-implicitly-destructible types).

  The `AnyType`, `Movable`, and `Copyable` traits no longer require that a type
  provide a `__del__()` method that may be called by the compiler implicitly
  whenever an owned value is unused. Instead, the `ImplicitlyDestructible` trait
  should be used in generic code to require that a type is implicitly
  destructible.

  Linear types enable Mojo programs to encode powerful invariants in the type
  system, by modeling a type in such a way that a user is required to take an
  action "in the future", rather than simply implicitly dropping an instance
  "on the floor".

  Code using `T: AnyType` can change to use `T: ImplicitlyDestructible` to
  preserve its pre-existing behavior following this change.

  Relatedly, the `UnknownDestructibility` trait is now no longer required, as it
  is equivalent to the new `AnyType` behavior.

- The `__next_ref__` method in for-each loops has been removed.  Now you can
  implement the `__next__` method of your iterator to return either a value or a
  reference.  When directly using the collection, Mojo will use the
  ref-returning variant, but will allow it to conform to `Iterator` for use with
  generic algorithms (which use a copied value).

- The `origin_of(x)` operator now returns a value of type `Origin` instead of an
  internal MLIR type, and aliases like `ImmutOrigin` are now `Origin` type as
  well.

- The `Origin.cast_from[x]` syntax has been replaced with a safe implicit
  conversion from any origin to an immutable origin (`ImmutOrigin(x)`) and an
  explicit unsafe conversion (`unsafe_origin_mutcast[origin, mut=m]`).

- Mojo no longer supports overloading functions on parameters alone: it will not
  try to disambiguate between `fn foo[a: Int8]():` and `fn foo[a: Int32]():` for
  example.  Mojo never fully implemented the previous support in a reliable way,
  and removing this simplifies the language.  It still supports overloading on
  function arguments of course.

### Library changes

- `PythonObject` now supports implicit conversion from `None`, allowing more
  natural Python-like code:

  ```mojo
  var obj: PythonObject = None  # Now works without explicit PythonObject(None)

  fn returns_none() -> PythonObject:
      return None  # Implicit conversion
  ```

- `IndexList` is no longer implicitly constructible from `Int`. Previously, the
  fill constructor (which broadcasts a single `Int` to all elements) was marked
  `@implicit`, allowing code like `var x: IndexList[3] = 5` which would create
  `(5, 5, 5)`. This implicit conversion has been removed to improve type safety.
  Use explicit construction instead: `IndexList[3](5)`.

- The `inlined_assembly` function is now publicly exported from the `sys` module,
  allowing users to embed raw assembly instructions directly into Mojo code.
  This provides fine-grained control over hardware operations using LLVM-style
  inline assembly syntax. Example:

  ```mojo
  from sys import inlined_assembly

  # Convert bfloat16 to float32 on NVIDIA GPU using PTX assembly.
  var result = inlined_assembly[
      "cvt.f32.bf16 $0, $1;",
      Float32,
      constraints="=f,h",
      has_side_effect=False,
  ](my_bf16_as_int16)
  ```

- We have removed `Identifiable` from enum-like types
  (such as `DType` and `AddressSpace`). This change is
  related to the idea that `Identifiable` is for comparing memory addresses.

- The `Iterator` trait and and for-each loop have removed the `__has_next__`
  method and now using a `__next__` method that `raises StopIteration`. This
  follows Python precedent better, is more convenient to implement, and can be a
  minor performance win in some cases.

- `Variadic` now has `zip_types`, `zip_values`, and `slice_types`.

- The `reflection` module has been moved from `compile.reflection` to a top-level
  `reflection` module. Update imports from `from compile.reflection import ...`
  to `from reflection import ...`.

- The `reflection` module now supports compile-time struct field introspection:

  - `struct_field_count[T]()` returns the number of fields
  - `struct_field_names[T]()` returns an `InlineArray[StaticString, N]` of
    all field names
  - `struct_field_types[T]()` returns a variadic of all field types
  - `struct_field_index_by_name[T, name]()` returns the index of a field by name
  - `struct_field_type_by_name[T, name]()` returns the type of a field,
    wrapped in a `ReflectedType` struct

  These APIs work with both concrete types and generic type parameters,
  enabling generic serialization, comparison, and other reflection-based
  utilities.

  Example iterating over fields (works with generics):

  ```mojo
  fn print_fields[T: AnyType]():
      comptime names = struct_field_names[T]()
      comptime types = struct_field_types[T]()
      @parameter
      for i in range(struct_field_count[T]()):
          print(names[i], get_type_name[types[i]]())

  fn main():
      print_fields[Point]()  # Works with any struct!
  ```

  Example looking up a field by name:

  ```mojo
  comptime idx = struct_field_index_by_name[Point, "x"]()  # 0
  comptime field_type = struct_field_type_by_name[Point, "y"]()
  var value: field_type.T = 3.14  # field_type.T is Float64
  ```

- Two new magic functions have been added for index-based struct field access:

  - `__struct_field_type_at_index(T, idx)` returns the type of the field at the
    given index.
  - `__struct_field_ref(idx, ref s)` returns a reference to the field at the
    given index.

  Unlike `kgen.struct.extract` which copies the field value, `__struct_field_ref`
  returns a reference, enabling reflection-based utilities to work with
  non-copyable types:

  ```mojo
  struct Container:
      var id: Int
      var resource: NonCopyableResource  # Cannot be copied!

  fn inspect(ref c: Container):
      # Get references to fields without copying
      ref id_ref = __struct_field_ref(0, c)
      ref resource_ref = __struct_field_ref(1, c)

      print("id:", id_ref)
      print("resource:", resource_ref.data)

      # Mutation through reference also works
      __struct_field_ref(0, c) = 42
  ```

  The index can be either a literal integer or a parametric index (such as a
  loop variable in a `@parameter for` loop), enabling generic field iteration:

  ```mojo
  fn print_all_fields[T: AnyType](ref s: T):
      comptime names = struct_field_names[T]()
      @parameter
      for i in range(struct_field_count[T]()):
          print(names[i], "=", __struct_field_ref(i, s))

  fn main():
      var c = Container(42, NonCopyableResource(100))
      print_all_fields(c)  # Works with any struct!
  ```

  This enables implementing generic Debug traits and serialization utilities
  that work with any struct, regardless of whether its fields are copyable.

- The `conforms_to` builtin now accepts types from the reflection APIs like
  `struct_field_types[T]()`. This enables checking trait conformance on
  dynamically obtained field types:

  ```mojo
  @parameter
  for i in range(struct_field_count[MyStruct]()):
      comptime field_type = struct_field_types[MyStruct]()[i]
      @parameter
      if conforms_to(field_type, Copyable):
          print("Field", i, "is Copyable")
  ```

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

- `Dict` now raises a custom `DictKeyError` type on failure, making lookup
  failures more efficient to handle.

  ```mojo
  var d = Dict[String, Int]()
  var key = "missing_key"
  try:
      _ = d[key]
  except e:
      print(e)  # Prints: DictKeyError
  ```

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

- Basic support for linear types in the standard library is now available.
  Linear types, also known as non-implicitly-destructible types, are types that
  do not define a `__del__()` method that the compiler can call automatically
  to destroy an instance. Instead, a linear type must provide a named
  method taking `deinit self` that the programmer will be required to call
  explicitly whenever an owned instance is no longer used.

  The updated `AnyType` trait can be used in parameters to denote
  generic code that supports object instances that cannot be implicitly
  destroyed.

  - `Span`, `UnsafePointer`, `Pointer`, and `OwnedPointer` can point to linear
    types.
    - Added `UnsafePointer.destroy_pointee_with()`, for destroying linear types
      in-place using a destructor function pointer.
  - `List`, `InlineArray`, `Optional`, `Variant`, `VariadicListMem`, and
    `VariadicPack` can now contain linear types.
    - `Variant.take` now takes `deinit self` instead of `mut self`.
    - Added `Variant.destroy_with` for destroying a linear type in-place with an
      explicit destructor function.
    - The `*args` language syntax for arguments now supports linear types.
  - `Iterator.Element` no longer requires `ImplicitlyDestructible`.
  - `UnsafeMaybeUninitialized` can now contain linear types.

- Using a new 'unconditional conformances' technique leveraging `conforms_to()`
  and `trait_downcast()` to perform "late" element type conformance checking,
  some standard library types are now able to conform to traits that they could
  not previously. This includes:

  - `List` now conforms to `Equatable`, `Writable`, `Stringable`,
    and `Representable`.
  - `Dict` now conforms to `Writable`, `Stringable`, and `Representable`.
  - `Set` now conforms to `Writable`, `Stringable`, and `Representable`.
  - `Deque` now conforms to `Writable`, `Stringable`, and `Representable`.
  - `InlineArray` now conforms to `Writable`, `Stringable`, and `Representable`.
  - `Iterator` no longer requires its type to be `Copyable`.

  - The following types no longer require their elements to be `Copyable`.
    - `Iterator`
    - `Tuple`
    - `Variant`
    - `Optional`

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

- `DeviceContext.enqueue_function()` and
  `DeviceContext.enqueue_function_experimental()` now automatically infer
  `func_attribute` to `FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(shared_mem_bytes)`
  when `shared_mem_bytes` is specified but `func_attribute` is not, for NVIDIA GPUs
  with allocations > 48KB. This eliminates the need to specify the same shared memory
  size twice in many cases, reducing boilerplate and preventing mismatched values.
  On AMD GPUs or for allocations ≤ 48KB, explicit `func_attribute` values
  should be provided when needed.

- `StringLiteral.format` will now emit a compile-time constraint error if the
  format string is invalid (instead of a runtime error).

  ```mojo
  "Hello, {!invalid}".format("world")
  # note: constraint failed: Conversion flag "invalid" not recognized.
  ```

- `Counter` now conforms to `Writable`, `Stringable`, and `Representable`.

- The `iter.peekable` function has been added. This allows users to peek at
  the next element of an iterator without advancing it.

- The "LegacyUnsafePointer" type has been changed to take its mutability as a
  first inferred parameter without a default, rather than a later explicit
  parameter with a default value of true. We recommend moving off of this type
  as soon as possible, but to roughly emulate the prior behavior, try out:

  ```mojo
  from memory import LegacyUnsafePointer
  comptime UnsafePointer = LegacyUnsafePointer[mut=True, *_, **_]
  ```

- the `os.process` submodule has been added with utilities to spawn and
  wait on processes. These use `posix_spawn` and do not go through the
  system shell.

- `Writer` and `Writable` have been moved into a new `format` module and out of
  `io`. These traits are not directly related to binary i/o, but are rather
  closely tied to type/value string formatting.

- `Writer` has been reworked to only support UTF-8 data instead of arbitrary
  `Byte` sequences. The `write_bytes` method has been replaced with
  `write_string`.

  - In line with these changes, `String`'s `write_bytes` method has also been
    deprecated, and its initializer `__init__(out self, *, bytes: Span[Byte])`
    has had its keyword argument renamed to `unsafe_from_utf8`. This bring it
    more in line with the existing `StringSlice` constructors and explicitly
    states that construction from arbitrary bytes is inherently unsafe.

- `String` has had its UTF-8 guarantees strengthened.
  - It now has three separate constructors when converting raw bytes
  (`Span[Byte]`) to a `String`
    - `String(from_utf8=...)`: Raises an error if the bytes are invalid UTF-8
    - `String(from_utf8_lossy=...)`: Converts invalid UTF-8 byte sequences
      into the `(U+FFFD, �)` replacement character and does not raise an error.
    - `String(unsafe_from_utf8=...)`: Unsafely assumes the input bytes are valid
      UTF-8 without any checks.
  - `append_byte` has been deprecated and has been replaced with
    `append(Codepoint)`.

- `DeviceContext.enqueue_function_checked()` and
  `DeviceStream.enqueue_function_checked()` have been renamed to
  `enqueue_function()`. Similarly, `DeviceContext.compile_function_checked()`
  has been renamed to `compile_function()`.

- External origins are now expressed using type level
  `{Mut,Immut,}ExternalOrigin` aliases instead of being spelled like
  `Origin[True].external`, improving consistency with other origin types.

### Tooling changes

- The Mojo compiler now supports the `-Werror` flag, which treats all warnings
  as compilation errors. This is useful for enforcing stricter code quality
  standards, particularly in CI/CD pipelines. The flag works with the Mojo
  compiler tools (`mojo run`, `mojo build`, `mojo package`, `mojo doc`).
  When used with `--disable-warnings`, warnings are promoted to errors first,
  so the errors are not suppressed.
  - The counterpart `-Wno-error` flag disables treating warnings as errors.
    When both flags are specified, the last one wins.
- The `--validate-doc-strings` flag has been deprecated for `mojo doc` and
  removed from other tools (`mojo build`, `mojo run`, `mojo package`). Use
  `-Werror` instead to treat warnings as errors.
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
- Docstring validation now includes `comptime` aliases. The
  `--diagnose-missing-doc-strings` flag now checks that public aliases have
  properly formatted docstrings (summary ends with period, starts with capital
  letter). Parametric aliases are also checked for proper `Parameters:` sections.
- Docstring validation with `--validate-doc-strings` now emits an
  error when an `fn` function is declared to raise an error (`raises`) but it's
  missing a [`Raises`
  docstring](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/docstring-style-guide.md#errors)
  (previously it emitted only a warning). Because Mojo automatically
  treats all `def` functions as [raising
  functions](/mojo/manual/functions#raising-and-non-raising-functions), we don't
  enforce `Raises` docs for `def` functions (to avoid noisy false positives).
- The Mojo LSP server now debounces document updates to reduce CPU usage during
  rapid typing. Previously, every keystroke triggered a full document parse;
  now updates are coalesced with a 150ms delay, reducing parse frequency by
  10-50x during active editing.
- The Mojo compiler now supports the `--experimental-export-fixit` flag for
  `mojo build`, `mojo run`, and `mojo package`. This flag exports fix-its to a
  YAML file compatible with `clang-apply-replacements`, instead of applying them
  directly. This is useful for integrating Mojo's fix-it suggestions into
  external tooling workflows. The flag is mutually exclusive with
  `--experimental-fixit` (which applies fix-its directly).

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
  fn foo[dt: DType]() -> Int where dt == DType.int32:
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

- The DeviceContext `enqueue_function_unchecked` and `compile_function_unchecked`
  have been removed. Please migrate the code to use `enqueue_function` and
  `compile_function`.

- The `UnsafePointer.offset()` method is now deprecated. Use pointer arithmetic
  instead:

  ```mojo
  # Before
  new_ptr = ptr.offset(n)

  # After
  new_ptr = ptr + n
  ```

### 🛠️ Fixed

- `Codepoint.unsafe_decode_utf8_codepoint()` no longer returns `Codepoint(0)`
  (NUL) when passed an empty span. Instead, a `debug_assert` now enforces the
  requirement that the input span be non-empty, consistent with the function's
  existing safety contract.

- [Issue #5732](https://github.com/modular/modular/issues/5732): Compiler
  crash when using `get_type_name` with types containing constructor calls in
  their parameters (like `A[B(True)]`) when extracted via `struct_field_types`.
- [Issue #1850](https://github.com/modular/modular/issues/1850): Mojo assumes
  string literal at start of a function is a doc comment
- [Issue #4501](https://github.com/modular/modular/issues/4501): Incorrect
  parsing of incomplete assignment
- [Issue #4765](https://github.com/modular/modular/issues/4765): Parser
  accepts pointless var ref a = n binding form
- [Issue #5578](https://github.com/modular/modular/issues/5578): ownership
  overloading not working when used with `ref`.
- [Issue #5137](https://github.com/modular/modular/issues/5137): Tail call
  optimization doesn't happen for tail recursive functions with raises.
- [Issue #5138](https://github.com/modular/modular/issues/5138): Tail call
  optimization doesn't happen for functions with local stack temporaries.
- [Issue #5361](https://github.com/modular/modular/issues/5361): mojo doc
  crashes on alias of parametrized function with origin.
- [Issue #5618](https://github.com/modular/modular/issues/5618): Compiler crash
  when should be implicit conversion error.
- [Issue #5635](https://github.com/modular/modular/issues/5635): `Deque` shrink
  reallocation incorrectly handled empty deque with `capacity > min_capacity`.
- [Issue #5723](https://github.com/modular/modular/issues/5723): Compiler crash
  when using `get_type_name` with nested parametric types from `struct_field_types`.
- [Issue #5731](https://github.com/modular/modular/issues/5731): Compiler crash
  when using reflection functions on builtin types like `Int`, `NoneType`, or
