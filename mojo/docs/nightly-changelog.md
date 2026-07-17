---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- Mojo now support type inference from literals initializer.

  ```mojo
  var x : List[_] = [1, 2, 3]
  var x : List[_] = [1.0, 2.0, 3.0]
  ```

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

- Import resolution behavior has been made consistent. When resolving an import
  of a module or package, in any given directory the resolution in order of
  preference is: source packages; precompiled `.mojoc` files; source modules;
  legacy precompiled `.mojopkg` files.

  Previously the behavior was unspecified and would pick whichever matching
  name it found in the directory first.

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

- Mojo has improved its tracking of import locations and is now able to show
  where a package containing a diagnostic was first introduced into the program:

  ```text
  Included from /bug.mojo:2:
  Included from /foo/__init__.mojo:3:
  Included from /foo/nested_pkg/__init__.mojo:4:
  /foo/nested_pkg/my_module.mojo:1:5: note: candidate not viable: unexpected argument
  def bar(): pass
      ^
  ```

  Compared to the previous:

  ```text
  Included from /foo/nested_pkg/__init__.mojo:1:
  Included from /foo/nested_pkg/__init__.mojo:4:
  /foo/nested_pkg/my_module.mojo:1:5: note: candidate not viable: unexpected argument
  def bar(): pass
      ^
  ```

  Note that for precompiled packages (`.mojoc` files), locations *inside* the
  package are omitted. For example, the above would instead resemble the
  following for a precompiled `foo` package:

  ```text
  Included from /bug.mojo:2:
  /foo/nested_pkg/my_module.mojo:1:5: note: candidate not viable: unexpected argument
  def bar(): pass
      ^
  ```

  Note also that for brevity the compiler does not report where any `std`
  packages are pulled in as they're treated as privileged and implicitly
  imported into every module. This includes if the user explicitly imports all
  or part of the standard library themselves.

- `imm` is now the preferred spelling for the `read` argument and
  closure-capture convention. `read` still works but will soon be deprecated.

- A file can now import another identically named module or package, assuming
  it is found first during import resolution (#4534):

  ```mojo
  # -------- #
  # foo.mojo #
  # -------- #
  from foo import bar   # a package on the system

  bar()
  ```

## Language changes

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

- Intra-package accesses without explicit `import`s are now deprecated and will
  be removed in a future release:

  ```mojo
  package/
      __init__.mojo:
        # Exported or re-exported symbols
        def foo(): pass

      module1.mojo:
        # Module-defined symbol
        def bar(): pass

      module2.mojo:
        # Previously able to implicitly use either of the above symbols, e.g.,
        foo()
        module1.bar()
  ```

  With this change, `module2.mojo` above must explicitly import symbols from
  elsewhere in the package:

  ```mojo
  # module2.mojo

  from . import foo
  from . import module1

  foo()
  module1.bar()
  ```

- The `@explicit_destroy` decorator is no longer sufficient for a `struct` type
  to opt-out of `ImplicitlyDeletable` conformance.

  As before, by default all Mojo structs implicitly conform to
  `ImplicitlyDeletable`. Mojo now requires writing a constrained
  `ImplicitlyDeletable where ...` conformance to narrow or opt-out of that
  trait.

  This works both for types
  that are never `ImplicitlyDeletable` (`where False`) and for types that are
  non-ImplicitlyDeletable based on a non-trivial condition (`where <cond>`):

  ```mojo
  # no @explicit_destroy necessary
  struct NeverDeletable(
      ImplicitlyDeletable where False
  ):
      def destroy(deinit self):
          pass

  comptime assert not conforms_to(NeverDeletable, ImplicitlyDeletable)

  # no @explicit_destroy necessary
  struct Container[T: AnyType](
      ImplicitlyDeletable where conforms_to(T, ImplicitlyDeletable)
  ):
      var value: Self.T

  comptime assert conforms_to(Container[Int], ImplicitlyDeletable)
  comptime assert not conforms_to(Container[NonDeletable], ImplicitlyDeletable)
  ```

  Using `@explicit_destroy` without an argument error string is now an error, as
  it would have no effect or purpose.

  `@explicit_destroy("custom error")` can still be used to provide additional
  instruction to users when an instance cannot be deleted implicitly.

  This simplifies the language by replacing special decorator behavior with
  generalized struct conformance logic.

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

- The compiler now rejects newlines in the middle of certain statements, where
  they were previously permitted:

  - Between `def`/`struct`/`trait`/`comptime` keywords and the following
    identifier
  - Between the `async` and `def` keywords on function definitions
  - Anywhere in the midst of an `import` statement, save for parenthesized
    import lists.

- It is now possible to import modules & packages through regular directories
  using the same path-like syntax.

  For example, given the following structure:

  ```text
  dir
  └── nested_dir
      ├── module.mojo
      └── package
          └── __init__.mojo
  ```

  It is possible to import from the modules and packages inside the directories
  `dir` and `nested_dir`:

  ```mojo
  import dir.nested_dir.module

  from dir.nested_dir.package import foo
  ```

  Note that an import statement *resolving* to a directory cannot later be used
  for scoped lookups as if it were a module or package:

  ```mojo
  import dir

  dir.nested_dir.package.foo() # error
  ```

## Library stabilizations

- `trait ImplicitlyDeletable`
- `trait Movable`
- `trait Copyable`
- `trait ImplicitlyCopyable`

- List
  - `def __init__(out self)`
  - `def __init__(out self, *, capacity: Int)`
  - `def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):`
  - `def __del__(deinit self) where conforms_to(Self.T, ImplicitlyDeletable):`

  - ```def __getitem__[
        origin: Origin, //
      ](ref[origin] self, slice: ContiguousSlice) -> Span[Self.T, origin]:
    ```

- Span

## Library changes

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

- `List.capacity` is now a `capacity()` method instead of a public field. This
  keeps the allocated capacity out of the stable public field surface, since it
  should only change indirectly through operations like `append()`. Replace
  `my_list.capacity` with `my_list.capacity()`.

- Renamed `StaticConstantOrigin` to `ImmStaticOrigin`, to align with the
  `Imm`-prefixed spelling used for the other immutable origins. The old name
  is still available as a deprecated alias and will be removed in a future
  release.

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
  `MutUnsafeAnyOrigin` or `ImmUnsafeAnyOrigin` are now deprecated and emit a
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

- `Pointer` and `UnsafePointer` have had their `type` parameter renamed to `T`.

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

  Arguments are passed using the Mojo calling convention, which is correct
  for scalar and register-passable arguments. Multi-field struct arguments
  are rejected at compile time because the Mojo and C conventions can
  disagree on how aggregates are passed.

## Tooling changes

- Added a `--lld-path` CLI flag. This overrides the LLD path that Mojo uses.

- `mojo-lsp-server` no longer parses or type-checks code blocks inside
  docstrings by default. This checking rests on unstable foundations in the
  LSP server and was prone to failing, producing false-positive diagnostics
  unrelated to the code being edited, for little value in return. Pass
  `-check-docstrings` when launching `mojo-lsp-server` from the command line
  to re-enable the previous behavior. We plan to make this checking more
  robust and re-enable it by default over time.

## GPU programming

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

## Fixed

- [#6784](https://github.com/modular/modular/issues/6784),
  [#6434](https://github.com/modular/modular/issues/6434) - `math.sqrt` on
  `Float64` now works on NVIDIA GPU. It lowers to the IEEE correctly-rounded
  hardware sqrt (`sqrt.rn.f64`) instead of being rejected at compile time.
  NVIDIA has no approximate f64 sqrt, so the `Float32` fast path continues to
  use `sqrt.approx.ftz.f32`.

- [#6755](https://github.com/modular/modular/issues/6755) - Volatile loads are
  no longer removed when their results are unused.

- Type refinement from a `conforms_to()` guard now applies inside the branches
  of a ternary `exp1 if cond else exp2` used in a `comptime` context, matching
  the existing `comptime if` statement behavior. For example, this now compiles:

  ```mojo
  trait HasProperty:
      comptime property: Int

  comptime get_property_or[T: AnyType] =
      T.property if conforms_to(T, HasProperty) else 0
  ```

  Previously the true branch failed with `'AnyType' value has no attribute
  'property'` because `T` was not refined under the guard.

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
  `field_ref` to produce an invalid `kgen.struct.gep`. For example, this now
  compiles and prints `True`:

  ```mojo
  @fieldwise_init
  struct Inner(Equatable, RegisterPassable):
      var x: Int
      var y: Int

  @fieldwise_init
  struct Outer(Equatable, RegisterPassable):
      var inner: Inner

  def main():
      var o = Outer(Inner(1, 2))
      print(o == o)
  ```

- A method whose return type references a constrained `comptime` member (one
  declared with a trailing `where` clause) is now accepted when the method's
  own `where` clause discharges that member's constraint.

  ```mojo
  trait Operation:
      comptime Output: AnyType

      def operate(self) -> Self.Output: ...

  struct MyList[T: AnyType](Operation where conforms_to(T, Movable)):
      comptime Output: AnyType where conforms_to(Self.T, Movable) = Int

      def operate(self) -> Self.Output where conforms_to(Self.T, Movable):
          return Int(123)
  ```

- A method whose return type is a generic struct instantiated with a
  parameter that only satisfies the struct's declared trait bound via the
  method's own `where` clause (rather than via the parameter's own
  declaration) is now accepted, instead of spuriously rejecting the returned
  value as a different, unconvertible type.

  ```mojo
  struct Collection[T: AnyType](Movable):
      def foo(
          var self,
      ) -> Iter[Self.T] where conforms_to(Self.T, ImplicitlyDeletable):
          return Iter(self^)

  @fieldwise_init
  struct Iter[T: ImplicitlyDeletable]:
      var _collection: Collection[Self.T]
  ```

- A struct using `where False` to opt out of a builtin trait's implicit
  synthesis (e.g. `Movable where False`) no longer spuriously fails to
  compile when one of its fields also opts out of that same trait. For
  example, this now compiles:

  ```mojo
  struct One(Movable where False):
      pass

  struct Two(Movable where False):
      var y: One
  ```
