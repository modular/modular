---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- mojo now picks `Array` (instead of `List`) as the default type to construct
  from a list expression. E.g.,

  ```mojo
  var x = [1, 2, 3]
  # type_of(x) = Array[Int, 3]
  ```

- `where` clauses now accept an optional string-literal message, written
  `where (condition, "message")`. The message is included in the compiler
  diagnostic when the constraint fails, and is supported everywhere `where`
  clauses are allowed: trailing function and struct constraints, struct
  conditional-conformance clauses, and `alias`/`comptime` declarations.

  ```mojo
  def foo[sc: Int]() where (sc > 1, "scaling factor must be greater than 1"):
      ...
  ```

  Calling `foo[0]()` now reports the message in the note:

  ```plaintext
  note: constraint declared here evaluated to False, expected '(sc > Int(1))':
  scaling factor must be greater than 1
  ```

  The message must be a string literal; a non-literal message is
  reported as an error.

- Support for `lambda` expressions: anonymous, single-expression closures that
  desugar to a nested `def`. As in Python, the body is a single expression with
  no `return`; unlike Python, the arguments are parenthesized and typed like in
  a `def` signature — for example `lambda (x: Int) {} -> Int: x + 1`. The
  capture list `{…}` and return type may each be elided: an omitted capture list
  imm-captures the body's free variables (and is thin when there are none), and
  an omitted return type defaults to `None` — so the bare `lambda: expr` is
  valid when `expr` is `None`-typed. These are fixed defaults, not inference (a
  non-`None` body still needs an explicit `-> T`).

  A thin (capture-free) `lambda` is a function value, exactly like a `def`
  referenced by name. As such it:

  - binds to a `comptime`;
  - passes as a `thin` function-typed parameter;
  - decays to a `thin` function pointer in runtime positions.

  Referencing an enclosing function or struct parameter keeps it thin. Any
  other `lambda` is a closure instance — a runtime value with no function type,
  so it does none of the above:

  - one that captures;
  - one that writes an `{imm}`/`{mut}` capture convention, even capturing
    nothing;
  - one with unbound parameters of its own (`lambda [N: Int](…)`), bound at
    each call.

- Mojo supports an (internal only for now) feature known as *interior origins*,
  which allows collections to protect from a common class of memory unsafety
  problems. `List`, for example, now returns element references bound
  to an *interior origin* of the list instead of the whole-list origin, so an
  element reference is invalidated when the list is mutated (for example by
  `append()` or `pop()`). Code that holds an element reference across such a
  mutation is now correctly rejected by the lifetime checker instead of
  silently dangling after a reallocation:

  ```mojo
  var list = [1, 2, 3]
  ref elem = list[0]
  list.append(4)  # may reallocate, invalidating `elem`
  print(elem)     # error: use of invalidated interior reference
  ```

- Mojo now supports type inference from literal initializers:

  ```mojo
  var x: List[_] = [1, 2, 3]
  var y: List = [1.0, 2.0, 3.0]
  ```

- Mojo now supports `==` and `!=` for type equality checks, and `_type_is_eq`
  has been removed.

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
  def takes_them(var **kwargs: Int): ...
  def pass_them(var **kwargs: Int):
    takes_them(**kwargs^)
  ```

- Dynamic function pointers with unbound type parameters can now be called
  directly. The compiler infers parameters from the call arguments and
  specializes the callee before the indirect call. This capability only works
  with a limited set of parameters - those which are specialized to a single
  value. This notably enables origin parameters on runtime function calls,
  which can also be implicit from variadics:

  ```mojo
  var fp1: def(*Int) thin -> None
  var fp2: def[a: ImmOrigin](ref [a] x: Int) thin -> None
  ...
  fp1(1, 2)
  fp2(42)
  ```

- Struct fields are no longer allowed to hide `UnsafeAnyOrigin` within a
  struct. For example, this is no longer accepted:

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

- Method `self` parameters must now have type `Self`. Custom `self` types are
  now rejected unless the method is annotated with the (temporary)
  `@__allow_legacy_custom_self_type` decorator. Switch to a `where` clause
  instead.

  ```mojo
  struct Foo[T: AnyType]:
      # ERROR:
      def foo(self: Foo[Int]):
          ...
  # Migrate to
  struct Foo[T: AnyType]:
      def foo(self) where Self.T == Int:
          ...
  ```

- Added support for checking variadic type-list operands with `conforms_to()`.
  For example, a variadic parameter list can pass its type-list value directly:

  ```mojo
  def copy_variadic_elements[*Ts: AnyType](
      *args: *Ts
  ) where conforms_to(Ts.values, Copyable):
      pass
  ```

  To check several distinct standalone types against a trait, conjoin scalar
  checks, for example `conforms_to(T, Trait) and conforms_to(U, Trait)`.

- Mojo has improved its tracking of import locations and now shows where a
  package containing a diagnostic was first introduced into the program:

  ```text
  Included from /bug.mojo:2:
  Included from /foo/__init__.mojo:3:
  Included from /foo/nested_pkg/__init__.mojo:4:
  /foo/nested_pkg/my_module.mojo:1:5: note: candidate not viable: unexpected argument
  def bar(): pass
      ^
  ```

  For precompiled packages (`.mojoc` files), locations *inside* the package are
  omitted. For brevity, the compiler also does not report where `std` packages
  are pulled in, since they are implicitly imported into every module.

- `imm` is now the preferred spelling for the `read` argument and
  closure-capture convention. `read` still works but will soon be deprecated.

- Parametric "generator" types can now be spelled with a dedicated keyword
  instead of having to use MLIR syntax directly. This keyword is subject to
  change in the future as we get experience with it. An example is:
  `def foo[type: __generator_type[size: Int] SIMD[DType.uint8, size*2]](): ...`.

## Language changes

- `size_of` now returns the allocation size: the store size rounded up to the
  type's alignment, which is the stride between adjacent elements of an array
  of that type. This changes the result only for types whose store size is not
  a multiple of their alignment (e.g., structs + whose `@align(N)` exceeds
  their natural alignment) and fixes memory corruption when such types were
  used in `List` and other collections whose growth copies `count * size_of`
  bytes.

- Mojo now rejects function overloads that differ only in argument convention
  (`imm` vs `mut`).

- Predefined and reserved words (for example `class`, `del`, `match`, `yield`)
  can no longer be used as the name of a free function. Doing so now errors at
  the declaration instead of silently producing a function that could never be
  called.

- Declaring a variable by assigning to a fresh name inside a function body is
  deprecated, and now warns with a fixit that inserts `var`:

  ```mojo
  def sum_to(n: Int) -> Int:
      var total = 0  # previously: `total = 0`
      for i in range(n):
          total += i
      return total
  ```

  Every first assignment to a name warns, `:=` walrus targets and a bare `x: T`
  annotation included. Binding forms that already spell out how they bind are
  unaffected: `for` targets, `with ... as`, `except ... as`, comprehension
  targets, and the `_` discard.

- A bare `**kwargs` is now an error; write `var **kwargs` (a fixit inserts it),
  in function declarations and function types alike. `var` was already the only
  supported convention — the sole exception to arguments defaulting to `imm`,
  applied silently before — so semantics are unchanged.

- User-written structs must now explicitly declare closure-trait conformance
  in their inheritance list to satisfy a `def(...) -> ...` closure trait.
  Previously a struct with a compatible `__call__` was accepted implicitly
  (duck-typing). Declare the trait in the struct's inheritance list:

  ```mojo
  def apply[F: def(Int) -> Int](f: F, x: Int) -> Int:
      return f(x)

  struct Double(def(Int) -> Int):  # previously: `struct Double:`
      def __call__(self, x: Int) capturing -> Int:
          return x * 2

  _ = apply(Double(), 5)
  ```

  Conformance is checked at struct definition rather than deferred to the use
  site.

- The import system has been overhauled to make name resolution explicit and
  consistent:

  - Import resolution now follows a consistent preference order within a
    directory: source packages, then precompiled `.mojoc` files, then source
    modules, then legacy precompiled `.mojopkg` files. Previously the order
    was unspecified.

  - Relative imports must use `from` (`from . import foo`); the `import .foo`
    form is no longer accepted.

  - Absolute imports `import a.b.c` now bind all of `a`, `a.b`, and `a.b.c`
    into the scope, where previously only `a.b.c` was made available. Two
    related bugs are fixed: `import a` followed by `import a.b` no longer
    errors with "invalid redefinition of 'a'", and function-scoped dotted
    imports (`import a.b` inside a function body) now work.

  - An imported package's submodules are now only accessible when the
    package's `__init__.mojo` re-exports them (for example, with
    `from . import sub`). An absolute import of the submodule
    (`import pkg.submodule`) always works, bypassing the `__init__.mojo`.

  - Intra-package accesses without explicit `import`s are deprecated and will
    be removed in a future release. A module must now explicitly import
    symbols defined elsewhere in its own package:

    ```mojo
    # module2.mojo — uses foo() from __init__.mojo and module1.bar()
    from . import foo
    from . import module1

    foo()
    module1.bar()
    ```

  - Modules and packages can now be imported through regular (non-package)
    directories using the same path-like syntax, for example
    `import dir.nested_dir.module`. An import statement that *resolves* to a
    directory cannot itself be used for scoped lookups
    (`import dir` then `dir.nested_dir.module.foo()` is an error).

  - A standalone module can no longer import its own name (for example,
    `import util` inside `util.mojo`). Such an import could only resolve to
    the module itself, silently shadowing any same-named package on the
    search path. Modules inside packages are unaffected.

  - Importing functions with the same name from different modules, combining
    them into one overload set, is now deprecated and emits a warning; a
    future release will reject the second import. Import the name from a
    single module instead.

  - Wildcard imports now resolve latest first, textually: declarations
    imported last shadow earlier ones, including those implicitly imported
    from `std.prelude`.

  - Error diagnostics on failed imports are now emitted per import site,
    instead of once per module.

- The `@explicit_destroy` decorator is no longer sufficient for a `struct` type
  to opt out of `Deinitable` conformance. As before, all structs
  implicitly conform by default; to narrow or opt out, write a constrained
  `Deinitable where ...` conformance instead — `where False` for types
  that are never deletable, or a non-trivial condition:

  ```mojo
  struct NeverDeletable(
      Deinitable where False
  ):
      def destroy(deinit self):
          pass

  struct Container[T: AnyType](
      Deinitable where conforms_to(T, Deinitable)
  ):
      var value: Self.T
  ```

  Using `@explicit_destroy` without an error-string argument is now an error on
  both `struct` and `trait` declarations, since it has no effect; remove it.
  `@explicit_destroy("custom error")` can still be used to give users
  additional instruction when an instance cannot be deleted implicitly.

- The destructor dunder method should now be spelled `__deinit__`, for naming
  parity with `__init__`. The old `__del__` spelling still works but now
  emits a deprecation warning with a fix-it to rename it:

  ```mojo
  struct Example:
      def __deinit__(deinit self):
          pass
  ```

- `where` clauses inside a parameter list (for example,
  `[x: Int where x > 0]`) are no longer supported, following a period of
  deprecation. Use a trailing `where` clause after the signature instead:

  ```mojo
  # Old (no longer supported):
  # def foo[x: Int where x > 0]():

  # New:
  def foo[x: Int]() where x > 0:
      pass
  ```

- The compiler now rejects newlines in the middle of certain statements, where
  they were previously permitted:

  - Between `def`/`struct`/`trait`/`comptime` keywords and the following
    identifier
  - Between the `async` and `def` keywords on function definitions
  - Anywhere in the midst of an `import` statement, save for parenthesized
    import lists.

- Struct types are now Movable by default. To opt-out of always-on movability,
  either explicitly specify a conditionally Movable conformance using
  `Movable where <cond>`, or opt out of Movable conformance entirely using
  `Movable where False`.

## Library stabilizations
<!-- rumdl-disable MD013 -->

- `trait Deinitable`
- `trait Movable`
- `trait Copyable`
- `trait ImplicitlyCopyable`

- List
  - `def __init__(out self)`
  - `def __init__(out self, *, capacity: Int)`
  - `def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):`
  - `def __init__(out self, *, length: Int, fill: Self.T) where conforms_to(Self.T, Copyable):`
  - `def __del__(deinit self) where conforms_to(Self.T, Deinitable):`
  - `def reserve(mut self, capacity: Int):`
  - `def resize(mut self, length: Int, fill: Self.T) where conforms_to(Self.T, Copyable & Deinitable):`
  - `def __getitem__[origin: Origin, //](ref[origin] self, slice: ContiguousSlice) -> Span[Self.T, origin_of(self)._get_owned_interior["element"]]:`
  - `def __iadd__(mut self, var other: Self, /) where conforms_to(Self.T, Copyable):`
  - `def extend(mut self, var other: Self):`
  - `def __contains__[dtype: DType, //](self: Span[Scalar[dtype], _], value: Scalar[dtype]) -> Bool`
  - `def __contains__(self, value: Self.T) -> Bool where conforms_to(Self.T, Equatable)`
  - `def __getitem__(ref self, idx: Int, /) -> ref[_] Self.T:`
  - `def __eq__(self, other: Self, /) -> Bool where conforms_to(Self.T, Equatable):`

- Bool
- Span
  - `def __init__(out self):`
  - `def __init__(other: Span, out self: ImmSpan[other.T, other.origin]):`
- String
  - `def __init__(out self, data: StringLiteral, /):`
  - `def __init__(out self, *, from_utf8_lossy: Span[Byte, _]):`
  - `def __eq__(self, rhs: String) -> Bool:`
  - `def __eq__(self, other: StringSlice) -> Bool:`
  - `def __ne__(self, other: StringSlice) -> Bool:`
  - `def __getitem__(ref self, idx: Int, /) -> ref[self.origin, self.address_space] Self.T:`

- Optional
  - `def __init__(out self):`
  - `def __init__(out self, var value: Self.T) where conforms_to(Self.T, Movable):`
  - `def __bool__(self) -> Bool:`

- Array
  - `def __getitem_param__[idx: Int, /](ref self) -> ref[self] Self.T:`
  - `def __getitem__(ref self, idx: Int, /) -> ref[self] Self.T:`
  - `def unsafe_ptr[...](ref[origin, address_space] self) -> Pointer[...]:`

- ImmPointer
- MutPointer

<!-- rumdl-enable MD013 -->

## Library changes

- The second parameter of `SIMD` has been renamed from `size` to `length`, to
  match the `SIMDLength` type it is declared with and the `length` vocabulary
  the rest of the library uses for element counts:

  ```mojo
  var v = SIMD[DType.float32, length=4](1.0, 2.0, 3.0, 4.0)
  print(v[0], Int(v.length))
  ```

  Positional uses such as `SIMD[DType.float32, 4]` are unaffected. Reading the
  parameter as `v.size` still works but is deprecated and warns, so existing
  code keeps compiling while you migrate. Binding it by keyword as
  `SIMD[dtype, size=4]` is an error and must be updated.

  `ComplexSIMD`'s matching `size` parameter has been renamed to `length` as
  well, so the two types stay consistent.

- `external_call()` can now call C variadic functions. The new keyword-only
  `num_fixed_args` parameter gives how many of the first arguments are fixed
  arguments of the callee; the arguments after those are passed as variadic
  arguments:

  ```mojo
  # int open(const char *path, int oflag, ...);
  var path_str = path
  var fd = external_call["open", c_int, num_fixed_args=2](
      path_str.as_c_string_slice().unsafe_ptr(), c_int(flags), c_int(0o666)
  )
  ```

  Left at its `None` default, the callee is declared non-variadic, which
  miscompiles variadic calls on targets whose ABI passes variadic arguments
  differently from fixed ones. On ARM64 macOS, for example, the `open()` call
  above would silently create the file with unrelated permission bits. A count
  of `0` is distinct from `None`: it declares a callee whose every argument is
  variadic.

- Files opened through `open()` with mode `"w"`, `"rw"` or `"a"` no longer have
  their permissions rewritten to `0o666`. `open()` used to follow the `open(2)`
  call with an unconditional `fchmod(0o666)` to work around the dropped mode
  argument, and `fchmod()` ignores `umask` and applies to an existing file too.
  Now that the mode argument arrives, a newly created file is `0o666 & ~umask`
  (`0o644` under the common `umask` of `022`, matching Python) and an existing
  file keeps the permissions it already had.

- The `MutSpan` and `ImmSpan` aliases are now exported from the prelude, so
  they no longer need an explicit import from `std.collections`. This matches
  the `Mut`/`Imm` aliases for `Pointer`, which the prelude already exported.

- `StringSlice` has been renamed to `StringSpan`, matching other non-owning
  view types such as `Span`. `StringSlice` remains available as a `comptime`
  alias for the time being to ease transition to the new name.

  `StringSpan` has `MutStringSpan` and `ImmStringSpan` aliases, matching the
  `Mut`/`Imm` aliases already provided for `Span` and `Pointer`. The previous
  `MutStringSlice` and `ImmStringSlice` names remain available as compatibility
  aliases.

- `GPUInfo.vendor` has been removed. It duplicated `GPUInfo.api`, which
  identifies the vendor precisely (`"cuda"`, `"hip"`, `"metal"`, or a stdlib
  plugin's own API name) rather than collapsing every plugin accelerator into
  one enum value. Compare `api` instead:

  ```mojo
  comptime use_apple_path = ctx.default_device_info.api == "metal"
  ```

  `Vendor` itself remains, as the classifier behind
  `has_amd_gpu_accelerator()`, `has_nvidia_gpu_accelerator()` and
  `has_apple_gpu_accelerator()`.

- `ImplicitlyDestructible` has been renamed to `Deinitable`, for
  consistency with the `deinit` argument convention and the `__deinit__`
  spelling of the destructor. Both `ImplicitlyDestructible` and the
  intermediate `ImplicitlyDeletable` spelling remain available as deprecated
  aliases.

- `Span` now has a keyword-only `address_space` parameter (defaulting to
  `AddressSpace.GENERIC`), so a span can view memory in a non-default address
  space, such as GPU shared memory:

  ```mojo
  var smem = stack_allocation[
      32, Float32, address_space = AddressSpace.SHARED
  ]()
  var tile = Span[
      mut=True, Float32, MutUntrackedOrigin, address_space = AddressSpace.SHARED
  ](unsafe_ptr=smem, length=32)
  ```

  Address-only operations (indexing, slicing, `unsafe_ptr()`, `as_imm()`, and
  the SIMD search helpers) work in any address space and preserve it in their
  results. `fill()` also works in any address space when the element type is
  register passable, since such a value may cross an address-space boundary.
  The remaining element-copying operations (iteration, `copy_from()`, hashing,
  equality, and writing) are still restricted to the default address space,
  for any element type.

- `List`, `Span`, and `String`/`StringSlice` (`byte=`, `codepoint=`, and
  `grapheme=`) indexing with a contiguous (non-strided) slice now aborts on
  an invalid slice instead of silently clamping it. `start`/`end` must each
  be in `0` to the container's length (`len(container)` for `List`/`Span`,
  `self.byte_length()` for `byte=`, `self.count_codepoints()` for
  `codepoint=`, `self.count_graphemes()` for `grapheme=`), inclusive, with
  `start <= end`; a negative index is always invalid for a contiguous slice.

  ```mojo
  var lst: List = [1, 2, 3]
  lst[0:100]  # previously clamped to `lst[0:3]`; now aborts
  lst[3:1]    # previously returned an ill-defined result; now aborts
  lst[-1:]    # previously wrapped to the last element; now aborts
  lst[:-1]    # previously wrapped to `lst[0:2]`; now aborts

  var s = "hello"
  s[byte=0:100]       # previously clamped to `s[byte=0:5]`; now aborts
  s[byte=-1:]         # previously wrapped to the last byte; now aborts
  s[codepoint=0:100]  # previously clamped to `s[codepoint=0:5]`; now aborts
  s[grapheme=3:1]     # previously returned an ill-defined result; now aborts
  ```

  The common "all but the last element" idiom must now spell the end index
  explicitly, since negative indices are no longer supported. Note that
  `lst[0 : len(lst) - 1]` still aborts on an empty `lst` (`0 : -1`); guard
  with `max` if `lst` may be empty:

  ```mojo
  lst[:-1]                     # aborts
  lst[0 : len(lst) - 1]        # use this instead
  lst[: max(len(lst) - 1, 0)]  # ...or this, if `lst` may be empty
  ```

- Any integer scalar can now be constructed from an `Intable` value, not just
  `Int`. This makes taking a pointer's address as an unsigned integer work
  directly:

  ```mojo
  var x = 42
  var p = Pointer(to=x)
  var addr = UInt(p)  # previously required `UInt(Int(p))`
  ```

- `Bencher.iter()` now accepts a raising closure as a runtime argument, so a
  benchmark whose body raises can pass a closure with an explicit capture list
  instead of an `@parameter` closure. Prefer the unified closure form over the
  deprecated `@parameter` one.

  ```mojo
  def bench_add(mut b: Bencher) raises:
      var a = PythonObject(42)
      var c = PythonObject(10)

      @always_inline
      def call_fn() raises {var a, var c}:
          var r = a + c
          keep(r)

      b.iter(call_fn)
  ```

- `Array` is no longer `Defaultable`. Previously it conformed to `Defaultable`
  but attempting to actually default construct an `Array` would fail to compile.

- `Error` is now `ImplicitlyCopyable`, so re-raising a caught error no longer
  requires the transfer sigil:

  ```mojo
  try:
      might_fail()
  except e:
      print("logging error:", e)
      raise e  # previously an error: use `raise e^`
  ```

  A captured `StackTrace` is now reference counted, so copying an `Error` costs
  a reference count increment rather than duplicating the trace. `raise e^`
  still works and avoids the copy.

- `PythonObject` arithmetic, comparison, and membership operators now dispatch
  through CPython's abstract number, object, and sequence protocols (for
  example `PyNumber_Add`, `PyObject_RichCompare`, and `PySequence_Contains`)
  instead of a Python-level attribute lookup followed by a bound-method call.
  Together with the non-mutating operators now borrowing their operand rather
  than taking it by value, this is roughly 12x faster on the interop hot path
  (a tight `a + b` or `a < b` loop). It also follows standard Python operator
  semantics more closely, including reflected-operand fallback (`__radd__`,
  `__rmul__`, and so on) and the standard error messages for unsupported
  operations. An operation that no operand supports now raises `TypeError`,
  where previously it could yield the `NotImplemented` object as a value, and
  comparing mismatched types with `==` now returns `False` rather than a truthy
  `NotImplemented`.

- `InlineArray` has been renamed to `Array`, its first parameter from
  `ElementType` to `T`, and its second parameter from `size` to `length`. A
  temporary `InlineArray` comptime alias exists for adoption, and `.size`
  remains as a deprecated alias for `.length`. Update explicit
  `InlineArray[ElementType=..., size=N]` usages to `Array[T=..., length=N]`.

- Many raw-pointer APIs across the standard library now use a safe `Pointer`
  instead of an `UnsafePointer`:

  - `List.unsafe_ptr()`, `InlineArray.unsafe_ptr()`, and
    `UnsafeUnion.unsafe_ptr()`.
  - The `unsafe_ptr()` accessors of `Span`, `StringSlice`, `String` (plus
    `String.unsafe_ptr_mut()`), `StringLiteral`, and `CStringSlice`.
  - `Allocation.unsafe_ptr()`, `Allocation.unsafe_leak()`, and
    `OwnedPointer.unsafe_ptr()`.
  - `PythonObject.unsafe_get_as_pointer()`,
    `PythonObject.downcast_value_ptr()`, and
    `PythonObject.unchecked_downcast_value_ptr()`.
  - The AMD `sys.intrinsics.implicitarg_ptr()` intrinsic.
  - `DevicePointer.unsafe_ptr()` (`std.gpu.host`) and the `unsafe_ptr()`
    requirement of the `DevicePointerLike` trait.
  - The `capture_sizes` field of `CompiledFunctionInfo` (`std.compile`), now a
    safe `Pointer[UInt64]`.
  - The `Span(unsafe_ptr=..., length=...)` constructor, matching `Span`'s
    internal pointer field.

  The two pointer types share the same layout and convert implicitly, so most
  code is unaffected. Code that called an unsafe-only pointer operation
  directly on the result should switch to the ungated `unsafe_*` spelling, for
  example `ptr + i` becomes `ptr.unsafe_offset(i)` and `ptr[i]` becomes
  `ptr[unsafe_offset=i]`.

- `OwnedPointer.steal_data()`, `ArcPointer.steal_data()`, and
  `List.steal_data()` have been renamed to `unsafe_take_allocation()` and now
  return an owning `Allocation` instead of a raw pointer. The methods keep an
  `unsafe_` prefix because the elements are handed over still initialized:
  deallocating does not run their destructors. Recover the previous raw pointer
  with `unsafe_leak()`. `List.steal_data()` and `OwnedPointer.steal_data()`
  remain as `@deprecated` methods, while `ArcPointer.steal_data()` is removed
  outright: the reconstructing `ArcPointer(unsafe_from_raw_pointer=...)`
  constructor now takes a pointer to the control block (obtained from
  `unsafe_take_allocation().unsafe_leak()`) and no longer accepts the payload
  pointer that `steal_data()` handed out.

- `OwnedPointer.take()` has been renamed to `OwnedPointer.into_inner()`.
  The old name remains as a `@deprecated` method and will be
  removed in a future release.

- `Variant.take[T]()` and `Variant.unsafe_take[T]()` have been renamed to
  `Variant.unwrap[T]()` and `Variant.unsafe_unwrap[T]()`. The old names
  remain as `@deprecated` methods and will be removed in a future release.

- The `as_immutable()` method on `UnsafePointer` and the
  `get_immutable()` method on `Span`, `StringSlice`, and `UnsafePointer`
  have all been renamed to a single `as_imm()` method, embracing the
  shorter `imm` spelling for a consistent immutability API. The old
  names remain as `@deprecated` aliases and will be removed in a future
  release.

- Added
  [`runtime.initialize_runtime()`](/docs/std/runtime/asyncrt/initialize_runtime/),
  which initializes the Mojo runtime when Mojo code built as a shared library
  (`mojo build --emit shared-lib`) is called from a non-Mojo host program such
  as C or C++. In that situation no Mojo `main()` function runs, so the runtime
  was never initialized and parallel or asynchronous APIs such as
  `parallelize()` crashed. Call `initialize_runtime()` before using any
  runtime-dependent API; the call is idempotent, and a single call covers all
  threads in the process. See
  [Call a Mojo shared library from C or C++](/docs/tools/compilation/#call-a-mojo-shared-library-from-c-or-c)
  for details.

- Added `List.try_index()`, which returns the index of a value in a list (if
  present) without raising, and is comptime-compatible.

- When an unhandled error propagates out of `main` and no stack trace was
  collected, Mojo now prints a hint to set
  `MODULAR_DEBUG=stack-trace-on-error` to enable stack trace collection,
  rather than printing only the error message.

- `Optional` and `Variant` now accept element types that are not `Movable`.
  Their element types are now bounded by `AnyType`, with `Movable`, `Copyable`,
  and related conformances conditional on the element types. A non-`Movable`
  value can be stored in place with the new closure-based `init_with=`
  constructors and `Variant.set()` overload, which construct the value directly
  into storage (placement-new) rather than moving it. `deinit_with()` on both
  types also no longer requires `Movable`, so element types that are neither
  `Movable` nor `Deinitable` are fully usable:

  ```mojo
  @fieldwise_init
  struct Pinned(Movable where False):
      var value: Int

  def make() -> Pinned:
      return Pinned(7)

  var opt = Optional[Pinned](call=make)    # construct in place
  var v = Variant[Pinned, Int](call=make)
  v.set(call=make)                         # replace in place
  ```

- `Optional` no longer conforms to `Iterator`; it is now an `Iterable`
  collection of 0 or 1 elements. `for value in opt` and `for value in opt^` are
  unchanged, but code that used an `Optional` directly as an iterator (for
  example calling `next()` on it) no longer compiles and should iterate the
  `Optional` instead.

- Various datatypes have adopted interior origins (described under language
  enhancements above), including `List`, `Deque`, `Variant`, `String`, `Dict`,
  `LinkedList`, `OwnedPointer`, and `HostBuffer`. A reference or view into one
  of these containers now carries an interior origin, so one held across a
  mutation is rejected by the lifetime checker instead of silently dangling
  after a reallocation. For example, `HostBuffer.as_span()` now returns a
  `Span` bound to an interior origin of the buffer instead of the whole-buffer
  origin:

  ```mojo
  var buf = ctx.enqueue_create_host_buffer[DType.float32](4)
  var s = buf.as_span()
  buf[0] = 1.0    # mutates the buffer, invalidating `s`
  print(s[0])     # error: use of invalidated interior reference
  ```

- `BitSet` gained `test_range[bit_value: Bool, *, lo: Int, hi: Int]`, which
  efficiently tests that a bit range holds an expected value, and resizing
  constructors: the `resized_from:` keyword constructor zero-extends from a
  smaller set (and debug-asserts no set bits are dropped when shrinking),
  while a companion overload taking a `truncate_set_bits: ()` keyword
  argument truncates instead.

- Added `Tuple.consume_elements`, which moves each element out of a tuple into a
  caller-provided closure one at a time. Destructuring such as `a, b = t^`
  copies each element, so it cannot take apart a tuple whose elements are
  `Movable` but not `ImplicitlyCopyable`; `consume_elements` transfers ownership
  instead, mirroring `VariadicPack.consume_elements`.

  ```mojo
  var t = ([1, 2, 3], [4, 5, 6])  # `List` is not `ImplicitlyCopyable`

  @parameter
  def handler[idx: Int](var elt: t.element_types[idx]):
      print(len(elt))

  t^.consume_elements[handler]()
  ```

- More renames standardizing on `length` over `size`: `TypeList.size` is now
  `TypeList.length`, and `DeviceContextList` is renamed to
  `DeviceContextArray` with its `size` parameter now `length` (update explicit
  `DeviceContextList[size=N]` to `DeviceContextArray[length=N]`). Similarly,
  `List.resize()` and `List.shrink()` now take `new_length` instead of
  `new_size`, and the `value` argument of `List.resize()` is renamed to
  `fill`, matching `List`'s constructor. The old names remain as deprecated
  aliases where applicable.

- `Span`'s pointer-and-length constructor argument is renamed from `ptr` to
  `unsafe_ptr`, to flag that this construction path is memory-unsafe: the caller
  must ensure the pointer addresses at least `length` valid elements. Update
  `Span(ptr=..., length=...)` to `Span(unsafe_ptr=..., length=...)`.

- `List.capacity` is now a `capacity()` method instead of a public field. This
  keeps the allocated capacity out of the stable public field surface, since it
  should only change indirectly through operations like `append()`. Replace
  `my_list.capacity` with `my_list.capacity()`.

- Renamed `StaticConstantOrigin` to `ImmStaticOrigin`, to align with the
  `Imm`-prefixed spelling used for the other immutable origins. The old name
  is still available as a deprecated alias and will be removed in a future
  release.

- `range()` has been reworked:

  - The `Int`-based and `Scalar`-based range types are unified into a single
    `dtype`-parameterized family, now that `Int` is `Scalar[DType.int]`.
    `range()` with `Int` arguments behaves exactly as before. As part of
    this, `range(...).__len__()` always returns `Int`, and asserts when an
    unsigned range's element count exceeds `Int.MAX` rather than silently
    clamping or wrapping; use `bounds()`, whose upper bound is `None` in that
    case, for the size hint.
  - Floating-point iteration is now drift-free and reversible. Element `i` is
    computed as `fma(i, step, start)`, so forward and reverse iteration
    produce identical sequences across repeated calls and across any IEEE-754
    platform at the same floating-point width. Previously a step that was not
    exactly representable, such as `0.1`, could drift and yield an extra
    forward element that `reversed()` then dropped.
  - `reversed()` now works on typed ranges such as
    `reversed(range(Int16(1), 10, 2))`. The `ReversibleRange` trait gained an
    associated `ReversedType` iterator instead of hard-coding its
    `__reversed__()` return type, so every range flavor can conform and
    return its own reversed iterator.
  - Non-numeric element types (`Bool` and the narrow MX float formats) are
    now rejected at construction, and the one- and two-argument float ranges
    (`range(Float64(4.5))` and `range(Float64(0.5), Float64(3.0))`) are
    compile errors instead of infinite loops; use the three-argument stepped
    form.

- `repr()` of a scalar `SIMD` value (`size == 1`) now prints using its type
  alias instead of the verbose `SIMD[DType.<dtype>, 1](...)` form when the
  dtype has one. For example, `repr(UInt32(4))` is now `UInt32(4)` (previously
  `SIMD[DType.uint32, 1](4)`), and `repr(List[UInt](1, 2))` is now
  `List[SIMD[DType.uint, 1]]([UInt(1), UInt(2)])`. `size > 1` values, and
  scalar dtypes without an alias (such as `DType.bool`), keep the
  `SIMD[...]` form. This only affects `repr()`; `String(...)` / `print(...)`
  output is unchanged.

- Renamed the raw memory functions to make their unsafety explicit:
  `memmove`, `memset`, `memset_zero`, `memcmp`, `uninit_move_n`,
  `uninit_copy_n`, and `destroy_n` are now `unsafe_memmove`, `unsafe_memset`,
  `unsafe_memset_zero`, `unsafe_memcmp`, `unsafe_uninit_move_n`,
  `unsafe_uninit_copy_n`, and `unsafe_destroy_n`. The old names are deprecated
  and will be removed in a future release.

- Added `Dict.insert(key, value)` and `Dict.clear_with(destroy_func)`, with
  mirroring `Set.insert(element)` and `Set.clear_with(destroy_func)`, so a
  `Dict` or `Set` whose key, value, or element type is not
  `Deinitable` can be populated and cleared. Unlike
  `dict[key] = value`, `insert` does not destroy a displaced entry: it moves
  it out and returns it as an `Optional` for the caller to destroy.
  `clear_with` hands each entry to `destroy_func` and retains capacity:

  ```mojo
  var d = Dict[Int, Int]()
  var displaced = d.insert(1, 10)  # None — key 1 was absent
  displaced = d.insert(1, 20)      # the displaced (1, 10) entry
  ```

- `Dict.fromkeys(keys, value)` has been generalized from taking a `List` to
  accepting any iterable of keys. Both forms require the key and
  value types to be `Deinitable`.

- `Counter` can now be constructed from any iterable of values, not just a
  `List`, e.g. `Counter(["a", "a", "b"])` or `Counter(String("aaab").bytes())`.
  This replaces the previous `Counter(items: List[V])` constructor.

- By-reference `Dict` iteration (`for entry in dict`, `keys()`, `values()`,
  `items()`, and `reversed()`) no longer requires the key and value types to be
  `Deinitable`. These iterators only borrow references and never
  destroy an entry, so they now work on a `Dict` whose key or value type is not
  `Deinitable`. Consuming iteration (`for entry in dict^` and
  `take_items()`) still requires `Deinitable`, since it drops the
  entries it does not yield.

- `Span` has moved from `std.memory.span` to `std.collections.span`.

- `AddressSpace` has moved from `std.memory.pointer` to
  `std.memory.address_space`.

- The container backing variadic `**kwargs` has been renamed from
  `OwnedKwargsDict` to `StringDict`. `StringDict` no longer
  requires its value type `V` to be `Deinitable`. A keyword dictionary
  whose values are linear (non-`Deinitable`) is itself linear and must
  be torn down explicitly with the new `deinit_with(deinit_func)`, which hands
  each key and value to `deinit_func`. It also gained `insert(key, value)`
  (returns the displaced entry as an `Optional[DictEntry]` without destroying
  it) and `popitem()` (moves out and returns a whole entry), mirroring `Dict`.
  Operations that destroy a displaced value in place — `kwargs[key] = value` and
  the two-argument `pop(key, default)` — still require `V` to be
  `Deinitable`; use `insert`, `popitem`, or the single-argument
  `pop(key)` for linear values.

- `Coord` now conforms to `DevicePassable`, so a `Coord` embedded in a
  `DevicePassable` type (such as a `TileTensor`'s `Layout`) is encoded to the
  device through `Coord._to_device_type` instead of a raw field bit-copy, the
  same way `IndexList` already was.

- Added `copy_to_numpy_array` and `from_numpy_array` to the new `python.numpy`
  module for moving flat numeric data between Mojo `Span`/`List` and NumPy
  arrays without hand-written `ctypes` plumbing:

  ```mojo
  from std.python.numpy import from_numpy_array, copy_to_numpy_array

  var values: List[Float64] = [1.0, 2.0, 3.0]
  var array = copy_to_numpy_array(values)            # NumPy array (copies)
  var span = from_numpy_array[DType.float64](array)  # borrow array as a Span
  ```

  Both support the fixed-width numeric dtypes. `copy_to_numpy_array` copies its
  input into a new, independent array; `from_numpy_array` borrows the array's
  buffer zero-copy.

- `Int` is now an alias for `Scalar[DType.int]` and integer literals materialize
  to this `Scalar` type. Because of this some conversions have become more
  strict.

  A new `SIMDLength` type has been added for the width of `SIMD` itself and must
  be used when inferring a parameter based on a SIMD argument like so:

  ```mojo
  def frob[w: SIMDLength](v: SIMD[DType.int, w]): ...
  ```

  Alternatively the width can be unbound if you simply want to be parametric
  over any `SIMD` type:

  ```mojo
  def frob(v: SIMD[DType.int, _])
  ```

  The new `Int` should still be used in all other situations.

  This type was briefly named `SIMDSize` earlier in this nightly cycle;
  `SIMDSize` remains as a deprecated alias for `SIMDLength`.

- `chdir` has been added to the `std.os` module and an `fchdir` method has been
  added to `io.FileDescriptor`. These are wrappers for the corresponding POSIX
  functions.

- `TypeList.all_conforms_to()` is now implemented in terms of `conforms_to()`,
  which supports parameter-list operands like `Ts.values`. As a result,
  `all_conforms_to()` constraints preserve the same proof structure as direct
  `conforms_to(Ts.values, Trait)` constraints, so the compiler can use them in
  conditional conformance implication checks and type refinement.

  This means conditional conformances can rely on trait hierarchy
  relationships for an entire type parameter pack. Given a trait
  `JsonSerializable` that inherits from `Serializable`, a conditionally
  conforming type previously had to repeat the inherited condition; now the
  derived condition alone is enough for the compiler to prove the inherited
  conformance:

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

- `is_trivially_destructible()` has been renamed to
  `is_trivially_deletable()`. It now accepts any type (`T: AnyType`) instead
  of requiring `T: Deinitable`, returning `False` for non-`Deinitable`
  (linear) types.

- `List.insert()` and `LinkedList.insert()` no longer normalize negative
  indices. Mojo collections are moving away from negative indexing, so the
  valid index range is now `[0, len(self)]`; a negative index is out of bounds
  and aborts (checked when asserts are enabled).

- The `Reflected.field_type[name]` reflection member has been renamed to
  `Reflected.field[name]`, because it returns a chainable `Reflected` handle
  for the named field rather than the field's bare type, so the old name was
  not accurate. Retrieve the field's type from the handle's `.T` member, as in
  `reflect[T].field["x"].T`. A by-index dual, `reflect[T].field_at[idx]`, has
  also been added so a field's concrete type can be recovered while iterating
  fields by index (where the name is not available as a literal):

  ```mojo
  comptime y_type = reflect[Point].field_at[1]
  var v: y_type.T = 3.14  # y_type.T is the concrete field type
  ```

- `Array[T]` (the type formerly known as `InlineArray[T]`) no longer conforms to
  `ImplicitlyCopyable`, since it is not inherently cheap to copy. It continues
  to conform to `Copyable`.

- Several collection types now *conditionally* conform to `Deinitable`,
  conforming only when their element type does. This lets a collection hold
  non-`Deinitable` elements at all (previously such a collection failed
  to compile); a collection of non-deletable elements is itself linear and must
  be drained explicitly with the new `deinit_with()` method, which calls a
  closure on each element:

  ```mojo
  collection^.deinit_with(my_destroy_closure)
  ```

  For `Deinitable` element types — the common case — all of this is
  transparent, but generic code that takes one of these collections by value
  may now need `& Deinitable` added to its element bound so the
  collection can be dropped:

  ```mojo
  def foo[T: Movable & Deinitable, //](var arr: InlineArray[T, 3]):
      pass
  ```

  Affected types, and the operations that still require `Deinitable`
  elements:

  - `InlineArray`: no remaining restrictions.
  - `Deque`: element-destroying operations (`append`, `appendleft`, `extend`,
    `extendleft`, `insert`, `clear`, `remove`, and so on) and consuming
    iteration (`for x in deque^`).
  - `Dict`: element-destroying and key/value-copying operations
    (`__setitem__`, `setdefault`, `fromkeys`, `update`, `__or__`, `__ior__`,
    `pop`, `clear`) and consuming iteration, so a `Dict` with linear keys or
    values can currently be constructed and torn down but not populated or
    mutated.
  - `LinkedList`: only `clear` and consuming iteration, so a `LinkedList`
    with linear elements can be populated (`append`, `prepend`, `insert`,
    `extend`) and torn down. `LinkedList.insert()` also no longer raises on
    an out-of-range index; like `List.insert()`, it now aborts (checked when
    asserts are enabled).
  - `Tuple`: a tuple with a linear element must be torn down with
    `deinit_with()` or fully consumed with `consume_elements()`. Generic code
    that stores a `Tuple[*Ts]` with an unbounded pack may need
    `& Deinitable` on the pack bound.
  - `Set`: the element bound loosened from `KeyElement & Deinitable`
    to just `KeyElement`; element-mutating operations (`add`, `remove`,
    `discard`, `clear`) and consuming iteration still require deletable
    elements, so a `Set` with linear elements can be constructed and torn
    down but not populated.
  - `OwnedPointer[T]`: conforms only when `T` does; a linear `OwnedPointer`
    must be consumed explicitly with `into_inner()` (for a `Movable` `T`) or
    `unsafe_take_allocation()` rather than dropped implicitly.

  Consuming iteration is conditional through the `IterableOwned` conformance;
  generic code bounded on `IterableOwned` now rejects a non-conforming
  element type at the bound rather than failing later inside `__iter__()`.

- `InlineArray`'s element type bound loosened from `Movable` to `AnyType`, so an
  `InlineArray` can now hold a non-`Movable` element type. The `Movable`
  conformance is now conditional on the element: move construction (including
  list-literal construction such as `[a, b, c]`) requires a `Movable` element,
  while indexing, by-reference iteration, and destruction do not. Code that
  uses `Movable` element types is unaffected, since a `Movable` element still
  yields a movable array.

- `Optional` gained `deinit_assert_empty()`, which destroys an empty linear
  `Optional` without a caller-provided deinitializer, aborting in safe-assert
  builds if it is non-empty. `Optional.map()` and `Optional.and_then()` also
  now work when the element type is linear (not `Deinitable`): they
  move the contained value out and destroy the emptied `Optional` explicitly,
  so a linear value can be transformed and handed back to the caller.

- It is now possible to iterate over owned elements in `List`, `Dict`,
  `InlineArray`, `LinkedList`, and `Set` when the element type is not
  `Copyable`: the `IterableOwned` conformance on these collections now
  requires only `Movable & Deinitable`, dropping `Copyable`.

  ```mojo
  def iterate[T: Movable](var list: List[T]):
    # Consume elements
    for var x in list^:
        pass
  ```

- The implicit conversion constructors that cast an `UnsafePointer` to
  `MutUnsafeAnyOrigin` or `ImmUnsafeAnyOrigin` are now deprecated and emit a
  deprecation warning when used. `UnsafeAnyOrigin` is an unsafe escape hatch
  that silently extends unrelated lifetimes and disables exclusivity checking,
  so it should never be applied implicitly. Prefer keeping a concrete origin;
  if you must discard it, make the cast explicit with the
  `as_unsafe_any_origin()` method.

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
  to `dyn_coord[DType]()`. Now one can just write
  `var my_coord = coord[1, 2, 3]` to create a
  `Coord[ComptimeInt[1], ComptimeInt[2], ComptimeInt[3]]`.

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

- The Python binding APIs now use safe pointers: the `PyCFunctionFast`
  calling convention used by `PythonModuleBuilder.def_py_c_function()` for
  `METH_FASTCALL` callbacks declares its argument array as a
  `Pointer[PyObjectPtr, MutUntrackedOrigin]`, typed-self methods registered
  through `PythonTypeBuilder.def_method()` declare their self parameter as a
  `Pointer[Self]` (for example, `self_ptr: Pointer[mut=True, Self]`), and the
  extension argument helpers `check_and_get_arg()` and
  `check_and_get_or_convert_arg()` return a safe `Pointer`. The pointer types
  share the same layout, so the C ABI and behavior are unchanged; update the
  spellings in signatures and read borrowed arguments with
  `args[unsafe_offset=i]`.

- Iterating over a `String`, `StringSlice`, or `StringLiteral` now yields
  grapheme clusters by default. Their `__iter__()` and `__reversed__()` methods
  return a `GraphemeSliceIter`, so `for c in my_string:` produces what a user
  perceives as a single "character" on screen. The lower-level views remain
  available when you want them: `codepoints()` or `codepoint_slices()` for
  Unicode scalars, and `bytes()` for raw UTF-8 bytes.

- The `Equatable` trait now allows for positional-only implementations, and
  arguments on implementers no longer need to match the trait exactly.

- `Pointer` and `UnsafePointer` have had their `type` parameter renamed to `T`.

- The `UnsafePointer` pointee-lifecycle methods are deprecated in favor of
  unified replacements that work on any `Pointer`, so callers no longer need
  to wrap safe pointers in `MutUnsafePointer`:

  - `init_pointee_move()` and `init_pointee_copy()` become `unsafe_write()`:
    pass the value by move (`ptr.unsafe_write(value^)`) or as the `copy`
    keyword argument (`ptr.unsafe_write(copy=value)`).
  - `destroy_pointee()` and `destroy_pointee_with()` become
    `unsafe_deinit_pointee()`: call it with no arguments to destroy an
    `Deinitable` pointee, or pass a deinitializing closure to
    destroy a non-`Deinitable` pointee in place.
  - `init_pointee_move_from()` becomes `unsafe_write_move_from(src)`, which
    moves the value out of a source pointer into the uninitialized memory
    `self` points to (leaving the source uninitialized).

- `Pointer` gained explicit `unsafe_`-prefixed methods for operations that are
  individually unsafe — unchecked bounds, aliasing casts, moving or overwriting
  memory — rather than requiring the whole pointer to be typed unsafe:
  `unsafe_offset()`, `unsafe_load()`, `unsafe_store()`, `unsafe_strided_load()`,
  `unsafe_strided_store()`, `unsafe_gather()`, `unsafe_scatter()`,
  `unsafe_as_noalias()`, `unsafe_address_space_cast()`, and
  `unsafe_take_pointee()`. These methods work on any `Pointer`. The previous
  unprefixed names still work, but are now hidden from the generated docs and
  remain gated behind an unsafe pointer type; prefer the `unsafe_`-prefixed
  names going forward. Each method's docstring documents the exact `Safety:`
  requirements the caller must uphold.

- The unprefixed pointer methods that the `unsafe_`-prefixed names above
  replace — `__getitem__()`, `__add__()`, `__sub__()`, `__iadd__()`,
  `__isub__()`, `load()`, `store()`, `strided_load()`, `strided_store()`,
  `gather()`, `scatter()`, `bitcast()`, `address_space_cast()`,
  `take_pointee()`, and `free()` — now emit a deprecation warning when called.

- The pre-unification pointer aliases `UnsafePointer`, `MutUnsafePointer`,
  `ImmUnsafePointer`, `ImmutUnsafePointer`, and `OptionalUnsafePointer` are
  now deprecated in favor of `Pointer`, `MutPointer`, `ImmPointer`, and
  `OptionalPointer`. The two pointer types were unified some time ago; the
  old names only existed for source compatibility with code written before
  that unification, and now emit a deprecation warning when used. Update
  type annotations and constructor calls to use the `Pointer` family
  instead:

  ```mojo
  # Deprecated:
  var ptr: UnsafePointer[Int, MutUntrackedOrigin]

  # Use instead:
  var ptr: Pointer[Int, MutUntrackedOrigin]
  ```

- `Pointer` now supports subtracting two pointers to compute the signed
  distance between them in elements of the pointee type, via the new
  `offset_from()` method (analogous to Rust's `offset_from`). The `-`
  operator does the same. Unlike the other pointer-arithmetic operators,
  which produce a new pointer and stay gated behind an unsafe pointer type,
  subtracting two pointers returns an `Int` distance and is available on
  safe pointers too:

  ```mojo
  var ptr = alloc[Int32](4)
  var end = ptr + 3
  print(end - ptr)  # => 3
  print(ptr.offset_from(end))  # => -3
  ptr.free()
  ```

- `OwnedDLHandle.get_function` now returns a callable that keeps the owning
  handle alive while it runs, fixing a crash where the library could be
  `dlclose`d between symbol lookup and the call. Its parameter is now the
  return type instead of the full function-pointer type, and it raises if the
  symbol is missing (previously it aborted the process):

  ```mojo
  # Before:
  var sqrt = lib.get_function[def(Float64) abi("C") -> Float64]("sqrt")
  # After:
  var sqrt = lib.get_function[Float64]("sqrt")
  ```

- `OwnedDLHandle.get_function` and `OwnedDLHandle.call` now forward arguments
  using the C ABI rather than the Mojo calling convention, so structs can be
  passed and returned by value. Multi-field struct arguments are no longer
  rejected at compile time.

- `OwnedDLHandle.get_symbol` now returns a pointer that borrows the handle
  instead of one with an untracked origin, so the library can no longer be
  `dlclose`d while a resolved symbol is still live. The `_ = lib` keep-alive
  that used to be needed to avoid that is now unnecessary. The pointer's
  mutability follows the handle's, so a symbol read through an immutable
  handle is read-only.

- The `cstr_name` overload of `OwnedDLHandle.get_symbol` now takes a
  `CStringSlice` rather than a `Pointer[mut=False, Int8]`, so the
  nul-termination it requires is stated by the type instead of assumed. Drop
  the `unsafe_ptr()` after `as_c_string_slice()` when calling it.

## Tooling changes

- Crash reporting now defaults to the `telemetry.enabled` setting, so the two
  are enabled or disabled together unless overridden. Setting
  `crash_reporting.enabled` (or the `MODULAR_CRASH_REPORTING_ENABLED`
  environment variable) explicitly still takes precedence. Previously crash
  reporting was disabled by default in one initialization path and enabled by
  default in production builds in another.

- The `program.crash_reporting_enabled_invocation` telemetry event has been
  renamed to `program.initialized`. It is emitted once per process whenever
  telemetry is enabled and carries a `crash_reporting.enabled` attribute
  recording whether crash reporting was on for that session.

- Added a `--lld-path` CLI flag. This overrides the LLD path that Mojo uses.

- `mojo-lsp-server` no longer parses or type-checks code blocks inside
  docstrings by default. This checking rests on unstable foundations in the
  LSP server and was prone to failing, producing false-positive diagnostics
  unrelated to the code being edited, for little value in return. Pass
  `-check-docstrings` when launching `mojo-lsp-server` from the command line
  to re-enable the previous behavior. We plan to make this checking more
  robust and re-enable it by default over time.

- Added a `--fp-mode` CLI flag that controls floating-point behavior as a
  comma-separated list of items. The only supported feature now is `contract`,
  one of `fast` (default) or `off`. `contract=fast` is like Clang's
  `-ffp-contract=fast`: `a + b*c` can fuse into a fused multiply-add across
  statements and breaking strict IEEE compliance;
  `contract=off` disables contraction for stricter floating-point semantics.
  The same `contract=fast|off` item is also accepted in the `emission_option`
  of a `kgen.compile_offload` operation, to control contraction of an
  individual offload kernel.

- Failed imports are no longer cached and may be retried, e.g., in the REPL.

## GPU programming

- `Int` and `UInt` no longer conform to `DevicePassable` and can no longer be
  passed as arguments to GPU kernels (via `DeviceContext.enqueue_function` or
  `compile_function`). They are platform-sized index types whose bit width
  depends on the host, so passing them to an accelerator miscompiles when the
  host and device disagree on the width (for example a 64-bit host driving a
  32-bit GPU index domain). Use a fixed-width type — `Int32`, `Int64`,
  `UInt32`, or `UInt64` — for kernel scalar arguments and parameters, and
  convert back with `Int(...)` inside the kernel body if you need a platform
  `Int` there. A kernel that still takes a bare `Int`/`UInt` argument now fails
  to compile with: "Int and UInt are not passable to device kernels; use a
  fixed-width type such as Int32 or Int64 instead".

- Added programmatic Metal GPU frame capture in `std.gpu.host`:
  `_start_metal_trace_capture(ctx, path)` and `_end_metal_trace_capture(ctx)`
  bracket GPU work and write a `.gputrace` file for offline replay (requires
  `MTL_CAPTURE_ENABLED=1`). A `_set_metal_gpu_print_enabled(ctx, enabled)`
  toggle and the `MODULAR_DISABLE_METAL_GPU_PRINT` environment variable disable
  Metal `os_log` GPU print; print is also suppressed during a capture, which
  otherwise cannot be replayed.

- A bare `--target-accelerator` architecture (for example `gfx950` or `sm_90`)
  is now handled identically to its vendor-prefixed form (`amdgpu:gfx950`,
  `nvidia:sm_90`). Previously `has_amd_gpu_accelerator()`,
  `has_nvidia_gpu_accelerator()`, and `has_apple_gpu_accelerator()` only
  recognized the vendor-prefixed spelling, so code that specialized on them
  (such as warp-tiling parameters) could silently take the wrong path and fail
  a downstream `comptime` constraint. `amd:<arch>` is also now accepted as an
  alias for `amdgpu:<arch>`, mirroring the existing `nvidia:<arch>` prefix.

- The GPU `Vendor` type can now be imported from `std.sys`
  (`from std.sys import Vendor`). It remains importable from
  `std.gpu.host.info` for backward compatibility.

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

- `Atomic.compare_exchange()` now accepts a `weak` parameter, and requires
  `weak=True` to compile on Apple GPU targets: AIR exposes no strong
  compare-exchange primitive, so Metal only lowers the `weak` form. This is
  safe for the common case of a CAS-retry loop, since a spurious failure just
  costs one extra iteration. Previously any use of `compare_exchange()`,
  including helpers built on it like atomic scatter-reduce, failed to
  compile on Metal.

- Apple M5 `simdgroup_matrix` MMA now accepts FP8 (`float8_e4m3fn`,
  `float8_e5m2`) inputs with an F32 accumulator, alongside the existing
  F16/BF16/F32 and 8-bit integer types.

- Added `warp.match_any()` and `warp.match_all()`: `match_any()` returns, for
  each warp lane, the mask of lanes whose value has the same bits, and
  `match_all()` returns the warp's active-lane mask if every lane holds the
  same bits and 0 otherwise. They use NVIDIA's `match.any.sync` and
  `match.all.sync` instructions, a `readfirstlane` ballot fold on AMD, and a
  shuffle-based emulation on Apple Silicon GPUs.

- `warp.vote()` now works on Apple Silicon GPUs. Metal's AIR backend exposes no
  usable ballot intrinsic, so it emulates the ballot with an XOR-butterfly
  OR-reduction over `simd_shuffle_xor`, returning a 32-bit mask (or a
  `DType.uint64` mask whose upper 32 bits are always zero); NVIDIA and AMD are
  unchanged.

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

- `DeviceGraphBuilder.add_function` now covers every live
  `DeviceContext.enqueue_function` form, so any kernel launchable on a device
  context can also be recorded as a graph node:

  - Added an overload that takes the kernel as a compile-time parameter and
    compiles it automatically, so callers no longer need a separate
    `DeviceContext.compile_function` step:

    ```mojo
    def build(mut builder: DeviceGraphBuilder) raises {read}:
        _ = builder.add_function[kernel](
            42, grid_dim=1, block_dim=1, dependencies=[]
        )
    ```

  - Added overloads accepting a `DeviceExternalFunction` loaded from PTX/SASS
    via `DeviceContext.load_function()`, and a capturing kernel as a
    compile-time parameter with runtime arguments.
  - All `add_function` overloads now accept a `location` argument so wrappers
    can attribute launch errors to their callers, and the closure overload now
    accepts (and honors) a `func_attribute` argument.

- Some standard library APIs related to accelerator programming have moved to
  a new `max` Mojo package, including:

  - `std.benchmark.Bench.bench_multicontext` ->
    `max.benchmark.bench_multicontext`
  - `std.benchmark.Bencher.iter_custom(DeviceContext)` ->
    `max.benchmark.bencher_iter_custom`
  - `std.gpu.compute` -> `max.gpu.compute`
  - `std.gpu.host` -> `max.gpu.host`
  - `std.gpu.memory` -> `max.gpu.memory`
  - `std.gpu.sync` -> `max.gpu.sync`

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

- The `layout` package is now bundled with MAX instead of Mojo.

- The GPU device-side standard library now uses the unified safe `Pointer`
  type throughout `std.gpu` (`memory`, `compute`, `intrinsics`, `sync`, and
  `primitives`). Public signatures that previously took or returned
  `UnsafePointer` are respelled to bare `Pointer`; since `Pointer` and
  `UnsafePointer` share representation and origin and decay implicitly, this
  is a type-identical change for callers. One visible difference:
  `external_memory()` now returns a safe `Pointer` instead of an
  `UnsafePointer`. Code that performs raw pointer arithmetic on the result or
  builds a `LayoutTensor`/`TileTensor` from it can wrap it in an
  explicitly-typed `UnsafePointer[...]` at the call site.

## Removed

- Removed the deprecated `DeviceContext.compile_function_experimental()` and
  `DeviceContext.enqueue_function_experimental()` methods, along with overloads
  that passed the kernel twice. Use `DeviceContext.compile_function[func]()`
  and `DeviceContext.enqueue_function[func]()` instead.

- Removed the `UInt`-returning GPU indexing accessors (`thread_idx_uint`,
  `block_idx_uint`, `block_dim_uint`, `grid_dim_uint`, `global_idx_uint`,
  `lane_id_uint`, `warp_id_uint`). Use the `Int`-returning `thread_idx`,
  `block_idx`, `block_dim`, `grid_dim`, `global_idx`, `lane_id`, and
  `warp_id` accessors instead.

- Removed the `store_volatile()` and `load_volatile()` intrinsics from
  `std.gpu.intrinsics`. Use `UnsafePointer.store[volatile=True]()` and
  `UnsafePointer.load[volatile=True]()` instead, which work across all
  supported GPU targets rather than NVIDIA only.

- Removed the deprecated `GPUAddressSpace` alias for `AddressSpace`. Use
  `AddressSpace` directly.

- Removed the `DType.invalid` sentinel alias. Code that used it to represent an
  absent or optional dtype should use `Optional[DType]` instead. Accordingly,
  `DType._from_str()` now returns an `Optional[DType]` (`None` when the string
  does not name a dtype) rather than `DType.invalid`.

- Removed positional indexing on `StringLiteral` (`literal[i]`). It allowed
  out-of-bounds reads and was inconsistent with the `[byte=]`, `[codepoint=]`,
  and `[grapheme=]` indexing scheme used by `String` and `StringSlice`. Use
  those keyword accessors instead (for example, on a `StaticString`).

- Removed the static `String.write()` methods. Use the equivalent `String()`
  constructor instead, which accepts the same `Writable` arguments (for
  example, `String(a, b, sep=", ")` instead of `String.write(a, b, sep=", ")`).
  The member `write()` methods that append to an existing string are unchanged.

## Fixed

- Targeting an MI250X now works. While normalizing the architecture name,
  `gfx90a` was rewritten to the nonexistent `gfx90aa`, so both
  `--target-accelerator` and `GPUInfo.from_name` reported every spelling of the
  target (`gfx90a`, `mi250x`, `amdgpu:gfx90a` and `amd:gfx90a`) as an
  unsupported architecture.

- Code completion now reports the correct completion kind for names bound by a
  `from module import name` statement that hasn't been resolved yet. Structs,
  traits, and functions imported this way previously completed with no kind at
  all. Additionally, a renamed binding (`from module import name as
  other_name`) no longer disappears from the completion list when another
  binding to the same declaration is in scope.

- `debug_assert` generates less code, so builds with `-D ASSERT=all` compile
  faster. Calls with no message arguments no longer allocate a 2048-byte message
  buffer in the caller's frame, which previously grew with the number of asserts
  and could push GPU kernels past the stack frame limit. A no-message assert
  failure now reports `assertion failed` instead of an empty message.

- `debug_assert` has dedicated overloads for the no-message case, which generate
  less code and so compile faster than passing an empty message list.

- Code folding in VSCode now works for Mojo files. `mojo-lsp-server` no longer
  advertises folding-range support, which only produced docstring ranges and
  caused VSCode to disable its built-in indentation-based folding — leaving
  functions, structs, and blocks unfoldable. Editors now fall back to
  indentation-based folding until the server returns structural folding
  ranges.

- Fixed `print()` and `debug_assert()` emitting garbled output on AMD GPUs when
  a printed string's byte length was an exact multiple of 8. The AMDGPU
  `hostcall` printf interface reads each string up to its nul terminator, and
  the terminator was being dropped in that case, so the host read past the
  payload.

- `base64.b16decode` now raises on invalid input instead of silently producing
  corrupt output.

- Closures mixing `*args`, named keyword-only arguments, and `**kwargs` now
  all work as values. A capturing closure taking `**kwargs` no longer fails
  to compile ("no matching method in call to '_insert'"), and a call may now
  combine a `*` unpack, literal keyword arguments, and a `**` splat, as in
  Python: `f(*args, **kwargs^)` forwards both packed variadics directly, and
  `f(1, named=2, **kwargs^)` binds the literal keyword to its own named
  parameter alongside the splat. The reverse splat order
  (`f(**kwargs, *args)`) is rejected, matching Python, as is combining a `**`
  splat with other keyword arguments bound for the same `**kwargs`.

- [#6784](https://github.com/modular/modular/issues/6784),
  [#6434](https://github.com/modular/modular/issues/6434) - `math.sqrt` on
  `Float64` now works on NVIDIA GPU. It lowers to the IEEE correctly-rounded
  hardware sqrt (`sqrt.rn.f64`) instead of being rejected at compile time.
  NVIDIA has no approximate f64 sqrt, so the `Float32` fast path continues to
  use `sqrt.approx.ftz.f32`.

- [#4473](https://github.com/modular/modular/issues/4473) - The `offset`
  parameter of `FileHandle.seek()` (and `NamedTemporaryFile.seek()`) is now a
  signed `Int` instead of `UInt64`, so negative offsets relative to
  `os.SEEK_CUR` or `os.SEEK_END` work as the docstrings already showed.
  Previously a negative offset only compiled as a literal (via unsigned
  wrap-around) and could not be passed from a signed variable.

- [#6755](https://github.com/modular/modular/issues/6755) - Volatile loads are
  no longer removed when their results are unused.

- Type refinement from a `conforms_to()` guard now applies inside the branches
  of a ternary `exp1 if cond else exp2` used in a `comptime` context, matching
  the existing `comptime if` statement behavior. For example,
  `T.property if conforms_to(T, HasProperty) else 0` now compiles; previously
  the true branch failed with `'AnyType' value has no attribute 'property'`
  because `T` was not refined under the guard.

- A `comptime` member with a trailing `where` clause is now accepted as a
  witness for a conditional trait conformance when the conformance constraint
  implies the member's constraint, for example:

  ```mojo
  trait StaticSize:
      comptime SIZE: Int

  struct Foo[size: Int = -1](StaticSize where size >= 0):
      comptime SIZE: Int where Self.size >= 0 = Self.size
  ```

- The reflection-based default `Equatable` implementation no longer fails to
  compile for single-element `RegisterPassable` structs. Such a struct is
  flattened to its sole field's type, which previously caused the reflection
  `field_ref` to produce an invalid `kgen.struct.gep`.

- A method whose return type references a constrained `comptime` member (one
  declared with a trailing `where` clause) is now accepted when the method's
  own `where` clause discharges that member's constraint.

- A method whose return type is a generic struct instantiated with a
  parameter that only satisfies the struct's declared trait bound via the
  method's own `where` clause (rather than via the parameter's own
  declaration) is now accepted, instead of spuriously rejecting the returned
  value as a different, unconvertible type.

- A struct using `where False` to opt out of a builtin trait's implicit
  synthesis (for example, `Movable where False`) no longer spuriously fails
  to compile when one of its fields also opts out of that same trait.

- `CPython.PyCapsule_New` now takes its `name` argument as a `StaticString`
  instead of an owned `String`. CPython stores the `name` pointer directly in
  the capsule rather than copying it, so an owned `String` argument left the
  capsule holding a dangling pointer once the temporary was destroyed.

- A failed import no longer poisons its name for the rest of the compilation.
  Previously, after something like `import pkg.util` failed to resolve, a
  later `import util` would silently bind the cached failure even when a real
  `util.mojo` exists on the search path, making the module unimportable with
  no diagnostic.

- [#6485](https://github.com/modular/modular/issues/6485) - `Optional[T]` and
  `Variant[...]` no longer corrupt data for payload types that include a
  `Bool` field. The fix changes how unions are lowered to LLVM.

- Struct extensions are no longer imported onto structs which happen to share a
  name with their intended struct, when the extensions' intended struct is
  shadowed by another:

  ```mojo
  from pkg_a import *   # defines a Foo and extensions on it
  from pkg_b import Foo # defines another Foo and extensions on it
  ```

  Previously in the above example, the extensions defined by `pkg_a` would be
  imported and callable on the unrelated `Foo` struct imported from `pkg_b`.

- Importing a package whose name is a prefix of another package when split by
  dots no longer works:

  ```mojo
  # Used to import e.g., package_with.dots if it presented as a package:
  #   package_with.dots/
  #   └── __init__.mojo

  import package_with # now errors
  ```

- Importing escaped-identifier packages & modules whose names contain dots now
  works reliably.

  ```mojo
  from `package.with.dots`.`module.with.dots` import foo
  ```

  `mojo doc` and file-in-package builds also now use the whole dotted name for
  such packages, rather than truncating it at the first dot.

- Invalid SIMD vector lengths are now rejected during code generation.

- `mojo build` now links libm, so a program calling a math function implemented
  by it — `math.hypot`, `math.expm1`, and `math.tanh` on `Float64`, among
  others — builds successfully on Linux. Such a program previously ran fine
  under `mojo run` but failed to link, for example with `undefined reference to
  symbol 'hypot@@GLIBC_2.35'` followed by `libm.so.6: error adding symbols: DSO
  missing from command line`.
