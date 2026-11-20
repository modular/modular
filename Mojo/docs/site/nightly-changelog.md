---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- `if`, `comptime if`, and `__guard` now support semicolon-separated
  condition clauses that mix Boolean conditions and `case` patterns. Clauses
  evaluate left to right, and successful pattern bindings are available to
  later clauses and the successful path.

- Added the experimental `__guard` statement for early exits from Boolean
  conditions and `case` patterns. Its `else` body must not fall through, so
  bindings from a successful pattern remain in scope on the normal path. This
  is analogous to features in Rust and Swift, and also supports
  `comptime __guard`.

- `try` blocks now unify heterogeneous raised error types rather than selecting
  the first raised type. Mojo selects a common type when one exists, otherwise
  uses `Error`. It emits a compile-time error when the raised types cannot be
  unified.

- `with` statements now no longer copy when passed a reference to a context
  manager like in `with foo: pass`.

- `__exit__(self, error: E)` now additionally supports `var self` and
  `deinit self`, similar to the non-error `__exit__`.

- You can bind a non-`Movable` payload from an rvalue enum or tuple subject
  with a `var` case binding, enabling `case .Some(var value):` and
  `case (var x, var y):` without copies. This allows `match` and `if case`
  statements to work with non-movable and linear types, and reduces copies in
  general.

- You can destructure a tuple of transferred non-copyable values into a `var`
  pattern without copies, such as `var (x, y) = (a^, b^)`. This includes
  movable types that are not `Copyable`.

- Comparisons now try the reflected method on the right operand when the left
  operand does not support the comparison. For example, `0 < value` can call
  `value.__gt__(0)`.

- `match` now warns when literal cases do not include a catch-all pattern,
  helping identify unmatched values.

- A closure that captures a value with an `UnsafeAnyOrigin` in its type, such
  as a `Pointer[Int, MutUnsafeAnyOrigin]`, now produces a warning. A closure
  does not keep the value behind such a capture alive, so Mojo can destroy that
  value before the closure runs. Capture a value with a tracked origin instead.

- Auto-parameterization now reaches `_` holes nested inside other parameterized
  types, so an unbound parameter at any depth of an argument's type becomes an
  an implicit parameter of the function:

  ```mojo
  def process(p: Optional[Pointer[Int, _]]): ...
  ```

- Mojo now allows unbound a parameter that has a default dependencies, the
  default value will later be refined when the unbound parameter is provided:

  ```mojo
  def depends_on_a[a: Int]() -> Int:
      return a

  struct NestedDeps[a: Int, b: Int = depends_on_a[a]()]:
      ...

  # No longer a compilation error
  comptime Partial = NestedDeps[_]

  # Bound = NestedDeps[3, depend_on_a[3]()]
  comptime Bound = Partial[3]
  ```

- A call to a function that takes a variadic pack (`*args: *Ts`) can now mix
  arguments and unpacked packs. The callee receives one pack with all the
  elements in order:

  ```mojo
  def show[*Ts: Writable](*args: *Ts): ...

  def wrap[*Ts: Writable](*args: *Ts):
      show("[", *args, "]")
      show(*args, *args)
  ```

  Unpacking the same pack twice into a `mut` pack argument, such as
  `update(*args, *args)`, is an aliasing error.

- A conditional type `A if c else B` now refines to the selected arm inside a
  `comptime if c:` branch, its `else` branch, and after a `comptime assert c`.
  You can call the arm's methods and read its fields without a `rebind`:

  ```mojo
  def describe[present: Bool](value: Int if present else NoneType):
      comptime if present:
          print("value:", value + 1)  # `value` is an `Int` here
      else:
          print("empty")
  ```

  A value of the selected arm also converts to the conditional type, so you can
  return or pass it without a `rebind`:

  ```mojo
  def make[present: Bool]() -> Int if present else NoneType:
      comptime if present:
          return 42  # an `Int` converts to the return type
      else:
          return None
  ```

- The error for converting a function to an invalid function type now explains
  why the conversion cannot be made, such as a mismatch in argument counts,
  argument conventions, effects, etc:

  ```mojo
  def takes_read(f: def(Int) thin -> None):
      pass

  def mut_arg(mut x: Int):
      pass

  # error: ... cannot be converted from 'def mut_arg(mut x: Int) thin -> None'
  # to 'def(Int) thin -> None', argument 'x': expected 'imm' convention,
  # found 'mut'
  takes_read(mut_arg)
  ```

## Language changes

- `@explicit_destroy("<message>")` on a struct is deprecated and now emits a
  warning. Give the message on the struct's `Deinitable` conformance instead:

  ```mojo
  # Before
  @explicit_destroy("call `close()` instead")
  struct File(not Deinitable):
      ...

  @explicit_destroy("use `deinit_with()` for non-Deinitable elements")
  struct Box[T: Movable](Deinitable where conforms_to(T, Deinitable)):
      ...

  # After
  struct File(not Deinitable else "call `close()` instead"):
      ...

  struct Box[T: Movable](
      Deinitable where conforms_to(T, Deinitable) else (
          "use `deinit_with()` for non-Deinitable elements"
      ),
  ):
      ...
  ```

  The message on a conditional `Deinitable where <condition> else "<message>"`
  conformance is now reported when a value is abandoned for parameters that
  don't satisfy the condition. Previously, only an always-false conformance's
  message was reported.

- `@explicit_destroy("<message>")` on a trait is deprecated and now emits a
  warning. Only structs support a custom linear type error message.

- Precompiled packages now no longer leak names and provide the same surface as
  the same packages compiled from source. This may break your code (especially
  from `std`):

  ```mojo
  from std.os import rmdir

  def main():
    std.os.rmdir(...) # This no longer works - use rmdir, or import std.os.
  ```

- It is now possible to `import` a precompiled (`.mojoc`) package from a
  namespace, assuming it has been precompiled to be built at that namespace.
  See the help for `mojo precompile --namespace` for more information.

## Library stabilizations

<!-- rumdl-disable MD013 -->
- `ImmSpan` alias
- `MutSpan` alias
- `List.insert`
- `List.clear`
- `Span.__getitem__(self, slice: ContiguousSlice) -> Self:`
- `Span.__init__(out self, ref[Self.origin] list: List[Self.T]):`
- `Span.__init__(out self, ref[Self.origin, Self.address_space] array: Array[Self.T, _]):`
- `Span.__len__(self) -> Int:`
- `Hashable`
  - `Hashable.__hash__`
- `Hasher`
  - `Hasher.update`
  - `Hasher.finish`
- `Pointer.__init__(to=)`
- `Pointer.__eq__`
- `Pointer.__ne__`
- `Pointer.unsafe_bitcast`
<!-- rumdl-enable MD013 -->

## Library performance improvements

- Reduced the size of `Optional[T]` to the size of `T` when `T` is one of the
  following types, so code that stores them in an `Optional` uses less memory:

  - `Allocation`
  - `Layout`
  - `List`
  - `ManagedAllocation`
  - `Span`
  - `String`
  - `StringSpan`
  - `ThinAllocation`

  Don't depend on this size. It relies on bit patterns that each type's current
  representation never uses, and that representation could change in the
  future.

## Library changes

- The following enum-like types now use `snake_case` member names. The
  uppercase names remain as deprecated aliases:

  - `Ordering`: `not_atomic`, `unordered`, `relaxed`, `acquire`, `release`,
    `acquire_release`, and `sequential`.
  - `FastMathFlag`: `none`, `nnan`, `ninf`, `nsz`, `arcp`, `contract`,
    `afn`, `reassoc`, and `fast`. Match patterns, `String()`, and `repr()`
    use the new names too, so a pattern such as `case .CONTRACT:` must
    become `case .contract:`.

- An optional mutable pointer now implicitly converts to an optional
  immutable pointer, as `Pointer` does.

- Added `normpath()` to `std.os.path`, matching Python's `os.path.normpath()`.
  Contributed by [@BlueDestination](https://github.com/BlueDestination) in
  [#7342](https://github.com/modular/modular/pull/7342).

- Added `Path.devnull()`, which returns the path to the platform's null device
  (`/dev/null` on Linux and macOS), so code doesn't need to hardcode it.

- Added a builtin `filter()`, which takes an iterable and a predicate closure
  and yields the elements the predicate accepts:
  `filter(items, lambda (x: Int) -> Bool: x > 0)`.

- `take_while()` and `drop_while()` in `std.itertools` now take the predicate
  as an argument instead of a compile-time parameter, so it can be a closure
  that captures values. Write `take_while(items, pred)` instead of
  `take_while[pred](items)`.

- `List(length=, fill_with=)` and `Array(fill_with=)` now accept a function
  that raises. If it raises, the initializer destroys the elements it already
  wrote and propagates the error.

- `Span.binary_search_by()` no longer takes the comparator as a compile-time
  parameter. Pass it as an argument, `span.binary_search_by(cmp)`, which also
  lets it capture the value being searched for.

- `QuickBench.run()` in `std.benchmark` now takes a single closure with no
  arguments in place of its eleven overloads for functions of 0 to 10
  arguments. Capture the inputs instead of passing them separately:
  `qb.run(lambda () {x} -> Float32: exp(x), bench_id=BenchId("exp"))`.

- `ord()` and `Codepoint.ord()` on a runtime string (`StringSpan`, `String`)
  now raise an `Error` instead of aborting when the string is empty. They also
  raise when the string holds more than one codepoint; previously they
  silently returned the first codepoint unless assertions were enabled.
  Callers must now handle the error with `try` or a `raises` function. On a
  string literal, both never raise, and a literal that isn't exactly one
  codepoint is a compile-time error. Contributed by
  [@Tharun-Kumar-McW](https://github.com/Tharun-Kumar-McW) in
  [#7444](https://github.com/modular/modular/pull/7444).

  ```mojo
  var text = String("🔥")
  var codepoint = ord(text)  # 128293; raises if `text` isn't one codepoint
  comptime letter = ord("a")  # 97; never raises
  ```

- `pow()` now takes a modulus as a third argument, like Python's `pow()`:
  `pow(base, exp, mod)` computes `base ** exp % mod` for integers of up to 64
  bits without overflowing. The result has the sign of `mod`, and a negative
  `exp` uses the modular inverse of `base`, so `pow(3, -1, 7)` is `5`.

- `Tuple` element types no longer need to be `Movable`. A `Tuple` is now
  `Movable` only when all of its elements are.
  `Tuple` has a new `init_with` initializer, which constructs each element in
  place from a closure parameterized on the element index:

  ```mojo
  comptime T = Tuple[Pinned, Pinned]  # `Pinned` is not `Movable`.
  var t = T(init_with=lambda [i: Int]() -> T.Ts[i]: Pinned(i * 10))
  ```

- `Layout` now names its element count `length`, matching `Span` and `List`.
  Construct one with `Layout[T](length=n)` and read the count back with
  `len(layout)`: `Layout` and `ManagedAllocation` now conform to `Sized`. The
  `Layout(count=)` initializer and the `Layout.count()` method are deprecated.

  ```mojo
  var allocation = alloc(Layout[Int32](length=4))
  for i in range(len(allocation)):
      allocation.unsafe_ptr().unsafe_offset(i).unsafe_write(Int32(i))
  dealloc(allocation^)
  ```

- Raising an integral `SIMD` value (including `Int`) or a `SIMDLength` to a
  negative power now aborts at run time instead of silently returning `0`. Use
  a floating point base instead.

- Raising an `IntLiteral` to a negative `IntLiteral` power now computes the
  exact result as a float literal, the reciprocal of the positive power,
  instead of evaluating to `0`. The result is a float literal even when the
  base is `1` or `-1`, so `var x = (-1)**-3` is now a `Float64` holding
  `-1.0`. Zero to a negative power evaluates to `nan`.

  ```mojo
  var a = 2**-1             # 0.5, same as 1 / 2
  var b = Float64(2) ** -1  # 0.5
  var n = 2
  var c = n**-1             # aborts at run time
  ```

- Added `FileDescriptor` to `std.io`, an owning wrapper around
  `RawFileDescriptor`. It is a linear type: close it explicitly with
  `fd^.close()`, which raises if the close fails. It is movable but not
  copyable, and `Optional[FileDescriptor]` is the size of a single
  descriptor. `RawFileDescriptor` gains a raising `close()` method.
  `FileDescriptor(copy_borrowed=)` duplicates a descriptor with
  close-on-exec set, so child processes don't inherit the copy.

- `RawFileDescriptor.read_bytes()`, `RawFileDescriptor.fchdir()`,
  `RawFileDescriptor.close()`, `FileDescriptor.close()`, and
  `FileDescriptor(copy_borrowed=)` now raise `IOError`, so callers can
  match on the error's `kind()`. Code in a plain `raises` function still
  compiles unchanged.

- File and directory functions in `std.os`, such as `mkdir()`, `remove()`,
  and `listdir()`, now raise `IOError`, so you can check what went wrong
  with `e.kind()`:

  ```mojo
  try:
      os.remove(path)
  except e:
      if e.kind() != .not_found:
          raise e
  ```

- Added `BorrowedFileDescriptor` to `std.io`, a copyable, non-owning view
  of a `FileDescriptor` that carries the owner's origin, so it can be
  passed around without closing the descriptor or outliving its owner.

- `ArcPointer` element types no longer need to be `Deinitable`. Destroy an
  `ArcPointer` of a non-`Deinitable` type with the new `deinit_with()` method.

- Added an `OwnedPointer(init_with=)` initializer that constructs the value in
  place, so the element type doesn't need to be `Movable`, and an
  `OwnedPointer.deinit_with()` method that destroys an `OwnedPointer` whose
  element type isn't `Deinitable`. The initializers that copied a value into the
  allocation are now deprecated.

- `FileHandle` now owns its descriptor through a `FileDescriptor` and always
  refers to an open file. It is no longer `Defaultable`, and `close()`
  consumes the handle (`f^.close()`) instead of leaving it in an invalid
  state. It still raises if the close fails. Use `Optional[FileHandle]`,
  which is the same size as `FileHandle`, for a file that may not be open.
  `NamedTemporaryFile` follows the same rules: it always holds an open file,
  `close()` consumes it, and `Optional[NamedTemporaryFile]` is the same size
  as `NamedTemporaryFile`.

- The `OptionalPointer`, `OptionalMutPointer`, and `OptionalImmPointer`
  aliases are deprecated. Write `Optional[Pointer[T, origin]]`,
  `Optional[MutPointer[T, origin]]`, or `Optional[ImmPointer[T, origin]]`
  instead. An `_` origin nested inside the `Optional` makes the function
  generic over it, as it did with the aliases:

  ```mojo
  # Before
  def fill(buf: OptionalMutPointer[Byte, _]): ...

  # After
  def fill(buf: Optional[MutPointer[Byte, _]]): ...
  ```

- The `StringSlice`, `ImmStringSpan`, and `ImmStringSlice` aliases are
  deprecated. Use `StringSpan` instead. `StringSpan` is always immutable, so
  the `Imm` variants are redundant.

- `Span` now conforms to `Equatable` when its element type does.

- `stdin`, `stdout`, and `stderr` are now `BorrowedFileDescriptor` values,
  and the `file` argument of `print()` takes a `BorrowedFileDescriptor`
  instead of a `RawFileDescriptor`. You can pass a `FileHandle` or
  `FileDescriptor` directly, since both convert implicitly:

  ```mojo
  with open("log.txt", "w") as f:
      print("hello", file=f)
  ```

- `SIMD.try_cast()` converts between integer types, raising
  `TryCastSIMDError` if any element is out of range. `SIMD.cast()` wraps
  instead, so `Int(-1).cast[.uint8]()` is 255.

  ```mojo
  var byte = Int(255).try_cast[.uint8]()  # 255
  var checked = Int(-1).try_cast[.uint8]()  # raises TryCastSIMDError
  var lanes = SIMD[.int32, 4](1, 2, 3, 300).try_cast[.uint8]()  # raises
  ```

- `find()` and `rfind()` on `String`, `StringSpan`, and `StringLiteral` now
  return `Optional[Int]`, holding `None` when the substring isn't found,
  instead of `-1`. Use `x in s` to test for a substring, or bind the offset
  with a pattern:

  ```mojo
  var line = String("key=value")
  if case .Some(eq) = line.find("="):
      print(line[byte=:eq])  # key
  print("!" in line)  # False
  ```

  Ordered checks such as `s.find(x) >= 0` and arithmetic on the result no
  longer compile. Two old patterns still compile but now give the wrong
  answer, with no error or warning, so search your code for them:

  - `s.find(x) == -1` is always `False` and `s.find(x) != -1` is always
    `True`, because `-1` converts implicitly to `Optional[Int]`. This also
    applies when you store the result in a variable and compare it later.
  - `if s.find(x):` is now `True` only when `x` is found. Before, it was
    `True` when `x` was missing and `False` when `x` was at offset 0.

### Interior origins

More standard library APIs now return references, pointers, spans, and
iterators that carry an *interior origin* of the value they point into,
instead of the value's own origin. Reassigning the value, or calling a
`mut self` method on it, invalidates them. Code that uses one afterward is now
a compile-time error; before, it compiled and read freed or destroyed memory.
These APIs now return interior origins:

- `List.unsafe_ptr()`.
- `String.unsafe_ptr()`, `String.unsafe_ptr_mut()`, `as_c_string_span()`,
  `split_at_grapheme()`, and the `String` iterators, such as `codepoints()`
  and `bytes()`.
- `Optional.value()`, `Optional.unsafe_value()`, and `Optional.__getitem__()`.
- `OwnedPointer.ptr()` and `Variant.unsafe_get()`.
- The pointers and spans that `Allocation`, `ManagedAllocation`, and
  `ThinAllocation` return.

Most code that no longer compiles matches one of these patterns:

- A pointer held across a mutation, such as `append()`, that can reallocate.

  ```mojo
  var p = list.unsafe_ptr()
  list.append(4)
  print(p[])  # error: use of invalidated interior reference 'list["element"]'
  ```

- A reference used after its owner is reassigned.

  ```mojo
  ref i = opt.value()
  opt = None
  print(i)  # error: use of invalidated interior reference 'opt["value"]'
  ```

- A string mutated while one of its iterators is in use.

  ```mojo
  # error: use of invalidated interior reference 's["bytes"]'
  for c in s.codepoints():
      s += String(c)
  ```

## GPU programming

- Added the `apple-m6` and `apple-m6-metal4` accelerator targets, and
  `CompilationTarget.is_apple_m6()` to `std.sys`. Apple M6 GPUs are now
  detected automatically, so you no longer need to pass
  `--target-accelerator=apple-m5` on M6 hardware.

- `compile_options` now reports an error for LLVM codegen flags such as
  `frame-pointer=all`, which it accepted before without effect. The only
  codegen flag that had an effect, `denormal-fp-math-f32=<mode>`, still sets
  the f32 denormal mode of NVIDIA GPU kernels, and other targets now reject it.

- `Bool`, `Scalar` (including `Int`), `String`, `StringSpan`, and
  `StringLiteral` now conform to `ConvertibleToPython`, so they can be passed
  to generic code bounded on that trait. The dedicated `PythonObject`
  constructors for these types are removed; implicit conversion still works
  through the single `ConvertibleToPython` constructor.
  `ConvertibleToPython` gains an associated `ConversionToPythonErrorType`
  (default `Never`), so conversions that cannot fail, such as `Bool` and
  `Scalar`, stay non-raising, while `String` conversions keep raising
  `Error`. This is a breaking change. If a conversion is non-raising, the
  `raises` effect can be removed from the `ConvertibleToPython` constructor.
  If a conversion is raising, an error type must be assigned to
  `ConversionToPythonErrorType`. Contributed by
  [@christoph-schlumpf](https://github.com/christoph-schlumpf) in
  [#6547](https://github.com/modular/modular/pull/6547).

## Tooling changes

- `mojo build` and `mojo run` can now report what elaboration produced, to
  show which compile-time code makes a compile slow. `--elaboration-stats`
  lists each `comptime for` loop with its iteration count and the operations
  its unrolled iterations produced. It also lists the calls that compile-time
  evaluation made to the interpreter, summed by function and by call site,
  with their cache hits and the CPU time spent interpreting. The
  report has one part for each pipeline that elaborated: the host and each
  accelerator target. This option is hidden; use `--help-hidden` to list it.

  Like the timing reports, the report prints as text to stderr by default.
  `--timing-json` prints it as JSON instead, under an `elaboration` key, and
  `--timing-file` writes it to a file instead of stderr. If the compilation
  hits the cache, elaboration is skipped and nothing is reported. Clear
  compilation cache or point `MODULAR_CACHE_DIR` at an empty directory
  to get a full report.

- `mojo build` and `mojo run` have a new hidden `--parser-stats` option that
  reports the time the parser spends to resolve each declaration, to show
  which declarations make parsing slow. Each declaration's time excludes the
  resolutions it caused, so the times add up to the total parse time. The
  report uses the same output options as `--elaboration-stats`.

- `mojo precompile` may now take a `--namespace` option to allow it to bake the
  provided namespace into the package. A package `pkg` built with `--namespace
  foo.bar` must be placed at `foo/bar/pkg.mojoc` and imported as `foo.bar.pkg`.
  While precompiling, the package may refer to itself both by `pkg` and by
  `foo.bar.pkg`. Both paths resolve to the source being compiled.

## Removed

- Legacy closures are removed. The `capturing` function effect and the
  `@__parameter` (and `@parameter`), `@__copy_capture`, and
  `@__move_capture` decorators are now errors. Declare a nested function's
  captures in a capture list instead, and pass the closure as a value:

  ```mojo
  def apply[F: def(Int) -> Int](f: F, x: Int) -> Int:
      return f(x)

  def main():
      var offset = 10
      def add(x: Int) {var offset} -> Int:  # was @__copy_capture(offset)
          return x + offset
      print(apply(add, 1))
  ```

  A function passed as a compile-time parameter must now be `thin`, for
  example `[f: def(Int) thin -> Int]`.

- `CompiledFunctionInfo` no longer has `num_captures`, `capture_sizes`, or
  `populate` members. Kernels can no longer capture implicitly, so these
  were always empty.

- The AMD-only intrinsics `readfirstlane()`, `sendmsg()`, and
  `implicitarg_ptr()` moved from `std.sys.intrinsics` to `max.gpu.intrinsics`,
  next to the other vendor-specific GPU intrinsics. Import them from there
  instead. `std.sys.intrinsics.ballot()` is removed; use
  `max.gpu.primitives.warp.vote()`, which works on every GPU vendor.

- The `Powable` and `CeilDivableRaising` traits are removed. A type that
  defines `__pow__()` still supports the `**` operator without conforming to a
  trait, and `pow()` now takes `SIMD` values only. Nothing conformed to
  `CeilDivableRaising`, so the raising `ceildiv()` overload is removed too.

- Removed APIs that were deprecated in Mojo 1.1 or earlier. Use the
  replacement in each case:

  - The `UnsafePointer` alias and the `std.memory.unsafe_pointer` module that
    defined it. Use `Pointer`.

  - The `Pointer` methods `load()`, `store()`, `strided_load()`,
    `strided_store()`, `gather()`, `scatter()`, `free()`, `bitcast()`,
    `address_space_cast()`, and `take_pointee()`. Use their `unsafe_`-prefixed
    spellings, such as `unsafe_load()`.

  - `Pointer.mut_cast()`. Spell the mutability in the pointer type
    (`ImmPointer` or `MutPointer`), or use `unsafe_mut_cast()`.

  - Subtracting an integer from a `Pointer` and the `+=` and `-=` operators.
    Use `unsafe_offset()`. Adding an integer and positional indexing such as
    `ptr[i]` remain deprecated.

  - `memmove()`. Use `unsafe_memmove()`.

  - The `alloc[T](count)` overload. Use `alloc(Layout[T](length=count))`,
    which returns an `Allocation`, or `unsafe_alloc[T](count)` for a raw
    pointer.

  - `is_trivially_movable()`, `is_trivially_copyable()`, and
    `is_trivially_deletable()`. Use `IsTriviallyMovable[T]`,
    `IsTriviallyCopyable[T]`, and `IsTriviallyDeinitable[T]` from
    `std.traits`.

  - The `CStringSlice` alias and `as_c_string_slice()` on `String`,
    `StringSlice`, and `StringLiteral`. Use `CStringSpan` and
    `as_c_string_span()`.

  - `unsafe_ptr()` on `ArcPointer`, `OwnedPointer`, `CStringSpan`, and
    `StringLiteral`. Use `ptr()`.

  - `String.capacity()` and `String.reserve()`. Use `capacity_bytes()` and
    `reserve_bytes()`.

  - `Tuple.element_types` and `VariadicPack.element_types`. Use `Ts`.

  - The `AnyType` and `Deinitable` aliases in `std.builtin.anytype`, and the
    `Movable`, `Copyable`, and `ImplicitlyCopyable` aliases in
    `std.builtin.value`. These traits live in `std.traits` and are still
    exported from the prelude.

## Fixed

- `VariadicPack` now conforms to `Writable` when all of its element types do,
  so a pack can be passed to `print()`, `String()` and `repr()`.

- `find()` and `rfind()` on `String`, `StringSpan` and `StringLiteral` now
  honor `start` when the substring is empty. `"abc".find("", 2)` returns `2`
  instead of `0`, and both methods return `None` when `start` is past the end
  of the string. Contributed by [@dhruvilmehta](https://github.com/dhruvilmehta)
  in [#7451](https://github.com/modular/modular/pull/7451) and
  [@Antony-mcw](https://github.com/Antony-mcw) in
  [#7668](https://github.com/modular/modular/pull/7668).

- A bare `raise` in an `except` block no longer crashes the compiler when
  building with debug info.

- `std.ffi.c_pid_t` is now `c_int` instead of `Int`, matching the 32-bit C
  `pid_t` on Linux and macOS. As an `Int`, a `-1` returned by a function such
  as `waitpid()` read back as a large positive number, so checks for `-1` or a
  negative result never fired.

- On Apple GPUs, building a `SIMD` from device-buffer elements in a kernel, as
  in `SIMD[DType.float32, 2](src[i], src[i])`, no longer reads zeros.

- `SIMD.slice()` no longer passes `llvm.vector.extract` an offset that is not
  a multiple of the output width, which LLVM rejects. A wide slice at such an
  offset, like `slice[16, offset=8]()` on a 32-element vector, now takes the
  element-wise path, and aligned slices narrower than the target's native
  vector width, such as the halves `split()` returns for small vectors, now
  use the intrinsic.

- `**` with an integer exponent no longer truncates the exponent to 32 bits,
  so `Float64(2.0) ** Int64(4294967296)` is now `inf` instead of `1.0`, and
  `Float64(2.0) ** Int32.MIN` is now `0.0` instead of `1.0`.

- Fixed rejection of negative integer and floating-point literals in `match`
  patterns. Contributed by [@a-tarasyuk](https://github.com/a-tarasyuk) in
  [#7316](https://github.com/modular/modular/pull/7316).

- A `comptime if conforms_to(T, Deinitable)` guard now correctly refines `T`
  and deinits it after its last use.

- Building a `SIMD` value from a list literal with the wrong number of elements
  is now a compile-time error. Previously, `var x: Int = [1, 2, 3, 4]` compiled
  and kept only the first element.

- Printing a `Float32` or `Float64` now produces the shortest decimal that
  rounds back to the same value, with ties rounded to even. Previously about
  one `Float32` in 200 printed a decimal that parsed back to a neighboring
  value (for example, `33560532.0` printed as `33560530.0`), and values
  exactly halfway between two shortest decimals, such as `3.37890625`, rounded
  up instead of to even.

- `atol()` now rejects a number that ends in an underscore when whitespace
  follows it, as it already did without the whitespace. Previously
  `atol("9_ ")` returned `90`.
- `Process.run()` now passes the child process the arguments it was given.
  Arguments built at run time, such as `String("a") + String(i)`, reached the
  child as repeated copies of the last argument or as freed memory.

- Fixed a bug where a function that modified the elements of a `mut` variadic
  pack (`mut *args: *Ts`), or of a `VariadicPack` with mutable elements, could
  leave the values passed to it unchanged.

- `input()` no longer raises `EOF` after the first line when stdin is a pipe
  or redirected file.

- `std.sys.exit()` is now declared `-> Never`, so a function with a return
  type can end in a call to `exit()` without a trailing `return`.

- [#7339](https://github.com/modular/modular/issues/7339) - Mojo now keeps
  compilation caches that another installed Mojo or MAX still uses.
  MAX, the `mojo` CLI and projects on different Mojo versions can now share one
  cache directory. A process can still lose its cache if its Mojo install is
  replaced while it runs.

- `Dict`, `StringDict`, `Deque`, `LinkedList` and `List.insert()` now accept
  elements whose type names a mutable origin, such as
  `Pointer[Int, origin_of(x)]`, as `List.append()` already did. Inserting one
  was rejected as aliasing the collection.

- `FileHandle` can now read and write more than 2 GiB at once on macOS,
  instead of raising `Invalid argument` or aborting.

- [#7669](https://github.com/modular/modular/issues/7669) - At compile time,
  `unsafe_memmove()` is fixed. `List.insert()` evaluated at compile time no
  longer repeats the element at the insertion point, and `List.insert()` and
  `List.pop(i)` now work on lists such as `List[String]`.

- A NUL byte (`\0`) in the literal text of a t-string is now kept as text.
  Previously it ended the literal segment early, so `t"a\0b{x}"` with
  `x = "ls"` produced `alsb` instead of `a\0bls`.
