# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

- Parts of the Mojo standard library continue to be progressively open sourced!
  Packages that are open sourced now include:
  - `subprocess`

- Trait compositions are now supported via the `&` syntax. A trait composition
  combines two traits into one logical trait whose constraint set is the union
  of the constraint sets of the two original traits.

### Language changes

- The Mojo compiler now warns about obsolete use of `mut self` in initializers,
  please switch over to `fn __init__(out self)` instead.

- The syntax for adding attributes to an `__mlir_op` is now limited to inherent
  attributes (those defined by the op definition). Most users will not need to
  attach other kinds of attributes, and this helps guard against typos and mojo
  code getting outdated when the dialect changes.

- `def` functions now require type annotations on arguments, and treat a missing
  return type as returning `None`. Previously these defaulted to the `object`
  type which led to a variety of problems.  Support for `object` is being
  removed until we have time to investigate a proper replacement.

- The Mojo compiler now warns about stores to values that are never used, e.g.:
  `x = foo(); x = bar()` will warn about the first assignment to `x` because
  it is overwritten.  You can generally address this by deleting dead code, or
  by assigning to `_` instead: `_ = foo(); x = bar()`.  You may also encounter
  this in variable declarations, e.g. `var x = 0; ...; x = foo()`.  In this
  case, change the variable to being declared as uninitialized, e.g.
  `var x: Int`.  You may also silence this warning entirely for a variable by
  renaming it to start with an underscore, e.g. `_x`.

- Mojo can now use [user-declared `__merge_with__` dunder
  methods](https://github.com/modular/max/blob/main/mojo/proposals/custom-type-merging.md)
  to merge values if different types in ternary operations.  This has been
  adopted to allow pointers to work naturally with the ternary operator, for
  example `var x = one_pointer if cond else other_pointer`.

- Auto-parameterization now extends to struct metatypes. For example, this
  declaration `fn foo[M: __type_of(StringLiteral[_])]` will auto-parameterize
  on the unbound parameter of `StringLiteral`.

### Standard library changes

String types in Mojo got several significant improvements:

- The `String` type no longer copies data from `StringLiteral` and
  `StaticString` since they are known-static-constant values.  This allows us to
  make construction from these values be implicit, which improves ergonomics and
  performance together. It also implements the "small string optimization",
  which avoids heap allocation for common short strings.  On a 64-bit system,
  `String` can hold up to 23 bytes inline.

- The types `StringSlice` and `StaticString` are now part of the prelude, there
  is no need to import them anymore.  These are useful for code that just needs
  a "view" of string data, not to own and mutate it.

- The `StringLiteral` type has been moved to a more reliable "dependent type"
  design where the value of the string is carried in a parameter instead of a
  stored member. This defines away a category of compiler crashes when working
  with `StringLiteral` by making it impossible to express that.  As a
  consequence of this change, many APIs should switch to using `StaticString`
  instead of `StringLiteral`.

- `String` supports a new `String(unsafe_uninit_length=x)` constructor and
  `str.resize(unsafe_uninit_length=x)` for clients that want to allocate space
  that they intend to fill in with custom unsafe initialization patterns.  The
  `String(ptr=x, length=y)` constructor has been removed.

- `String` supports working with legacy C APIs that assume "nul" termination,
  but the details have changed: `String` is now no longer implicitly
  nul-terminated, which means that it is incorrect to assume that
  `str.unsafe_ptr()` will return a nul-terminated string.  For that, use the
  `str.unsafe_cstr_ptr()` method. It now requires the string to be mutable in
  order to make nul-termination lazy on demand. This improves performance for
  strings that are not passed to legacy APIs.

- The `List` type has been improved similarly to `String` to reduce
  inconsistency and enable power-user features, including removing adding
  `List(unsafe_uninit_length=x)` and `list.resize(unsafe_uninit_size=n)` methods
  avoid initialized memory that the caller plans to overwrite.

The following traits have been removed in favor of trait composition:
`EqualityComparableCollectionElement`.

- `Span` now has a `swap_elements` method which takes two indices and swaps them
   within the span.

- `Pointer` now has `get_immutable()` to return a new `Pointer`
  with the same underlying data but an `ImmutableOrigin`.

- You can now forward a `VariadicPack` that is `Writable` to a writer using
`WritableVariadicPack`:

```mojo
from utils.write import WritableVariadicPack

fn print_message[*Ts: Writable](*messages: *Ts):
    print("message:", WritableVariadicPack(messages), "[end]")

x = 42
print_message("'x = ", x, "'")
```

```text
message: 'x = 42' [end]
```

In this example the variadic pack is buffered to the stack in the `print` call
along with the extra arguments, before doing a single syscall to write to
stdout.

- Removed `unroll` utility. Now simply use `@parameter` on for-loops.

```mojo
from utils.loop import unroll

# Before
@always_inline
@parameter
fn foo[i: Int]():
    body_logic[i]()
unroll[foo, iteration_range]()

# After
@parameter
for i in range(iteration_range):
    body_logic[i]()
```

- The `is_power_of_two(x)` function in the `bit` package is now a method on
  `Int`, `UInt` and `SIMD`.

- The `constrained[cond, string]()` function now accepts multiple strings that
  are printed concatenated on failure, so you can use:
  `constrained[cond, "hello: ", String(n), ": world"]()` which is more comptime
  efficient and somewhat more ergonomic than using string concatenation.

- `pathlib.Path.write_text` now accepts a `Writable` argument instead of a `Stringable`
  argument. This makes the function more efficient by removing a String allocation.

- Added `pathlib.Path.write_bytes` which enables writing raw bytes to a file.

- Added `os.path.split_extension` to split a path into its root and extension.

- Added `os.path.is_absolute` to check if a given path is absolute or not.

- `PythonObject` is no longer implicitly constructible from tuple or list
  literals, e.g. `var x : PythonObject = [1, 2, "foo"]` is no longer accepted.
  Instead, please use named constructors like
  `var x = Python.list(1, 2, "foo")`.  We hope to re-enable the syntax in
  the future as the standard library matures.

- One can now specify the consistency used in atomic operations with the default
  being sequential consistency.

### GPU changes

- `debug_assert` in AMD GPU kernels now behaves the same as NVIDIA, printing the
thread information and variadic args passed after the condition:

```mojo
from gpu.host import DeviceContext

fn kernel():
    var x = 1
    debug_assert(x == 2, "x should be 2 but is: ", x)

def main():
    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel](grid_dim=2, block_dim=2)
```

Running `mojo run -D ASSERT=all [filename]` will output:

```text
At /tmp/test.mojo:5:17: block: [0,0,0] thread: [0,0,0] Assert Error: x should be 2 but is: 1
At /tmp/test.mojo:5:17: block: [0,0,0] thread: [1,0,0] Assert Error: x should be 2 but is: 1
At /tmp/test.mojo:5:17: block: [1,0,0] thread: [0,0,0] Assert Error: x should be 2 but is: 1
At /tmp/test.mojo:5:17: block: [1,0,0] thread: [1,0,0] Assert Error: x should be 2 but is: 1
```

- Removed deprecated `DeviceContext` methods `copy_sync` and `memset_sync`.

- Add `Variant.is_type_supported` method. ([PR #4057](https://github.com/modular/max/pull/4057))
  Example:

  ```mojo
    def takes_variant(mut arg: Variant):
        if arg.is_type_supported[Float64]():
            arg = Float64(1.5)
    def main():
        var x = Variant[Int, Float64](1)
        takes_variant(x)
        if x.isa[Float64]():
            print(x[Float64]) # 1.5
  ```

- The `type` parameter of `SIMD` has been renamed to `dtype`.

- The `Pointer.address_of(...)` and `UnsafePointer.address_of(...)` functions
  have been deprecated.  Please use the `Pointer(to=...)` and `UnsafePointer(to=...)`
  constructors instead.  Conceptually, this is saying "please
  initialize a `Pointer` (a reference, if you will) to *some other address in
  memory*.  In the future, these `address_of` functions will be removed.

- The `logger` package is now open sourced (along with its commit history)!
  This helps continue our commitment to progressively open sourcing more
  of the standard library.

### Tooling changes

- **Fixed SIMD boolean display in debugger:** SIMD boolean values now display
  correctly with proper bit extraction.

### Mojo Compiler

- The Mojo compiler is now able to interpret all arithmetic operations from
the `index` dialect that are used in methods of `Int` and `UInt` types.
That allows users to finally compute constants at compile time:

```mojo
alias a: Int = 1000000000
alias b: Int = (5 * a) // 2
```

previously compiler would throw error "cannot fold operation".

- New `--emit-llvm` option to the `mojo build` command that allows users to emit
LLVM IR. When `--emit-llvm` is specified, the build process will: compile mojo
code to LLVM IR, save the IR to a .ll file (using the same name as the input
 file), and print the IR to stdout for immediate inspection.

### ❌ Removed

- The `SIMD.roundeven()` method has been removed from the standard library.
  This functionality is now handled by the `round()` function.

- Error messages about the obsolete `borrowed` and `inout` keywords, as well as
  the obsolete `-> Int as name` syntax has been removed.

- The `StringableCollectionElement` trait has been removed in favor of
  `WritableCollectionElement`.

- The `object` type has been removed.

- `utils.numerics.ulp` has been removed.  Use the same `ulp` function from the
  `math` package instead.

### 🛠️ Fixed

- [#3510](https://github.com/modular/max/issues/3510) - `PythonObject` doesn't
  handle large `UInt64` correctly.

- [#3847](https://github.com/modular/max/issues/3847) - Count leading zeros
  can't be used on SIMD at compile time.

- [#4198](https://github.com/modular/max/issues/4198) - Apple M4
  is not properly detected with `sys.is_apple_silicon()`.

- [#3662](https://github.com/modular/max/issues/3662) - Code using `llvm.assume`
  cannot run at compile time.

- [#4273](https://github.com/modular/max/issues/4273) - `count_leading_zeros`
  doesn't work for vectors with size > 1 at comptime.

- [#4320](https://github.com/modular/max/issues/4320) - Intermittent
  miscompilation with bytecode imported traits.

- [#4281](https://github.com/modular/max/issues/4281) - MAX does not support RTX
  5000-series GPUs.

- [#4163](https://github.com/modular/max/issues/4163) - Corner case in
  initializers.

- [#4360](https://github.com/modular/max/issues/4360) - Fix constructor emission
  for parameterized types conforming to a trait composition.
