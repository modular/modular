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

### Standard library changes

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

- The types `StringSlice` and `StaticString` are now part of the prelude, there
  is no need to import them anymore.

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
  `var x = PythonObject.list(1, 2, "foo")`.  We hope to re-enable the syntax in
  the future as the standard library matures.

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

- The `Pointer.address_of(...)` function has been deprecated.  Please use the
  `Pointer(to=...)` constructor instead.  Conceptually, this is saying "please
  initialize a `Pointer` (a reference, if you will) to *some other address in
  memory*.  In the future, `Pointer.address_of(...)` function will be removed.

- The `logger` package is now open sourced (along with its commit history)!
  This helps continue our commitment to progressively open sourcing more
  of the standard library.

### Tooling changes

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

### 🛠️ Fixed

- [#3510](https://github.com/modular/max/issues/3510) - `PythonObject` doesn't
  handle large `UInt64` correctly.

- [#3847](https://github.com/modular/max/issues/3847) - Count leading zeros
  can't be used on SIMD at compile time.

- [#4198](https://github.com/modular/max/issues/4198) - Apple M4
  is not properly detected with `sys.is_apple_silicon()`.

- [#3662](https://github.com/modular/max/issues/3662) - Code using `llvm.assume`
  cannot run at compile time.
