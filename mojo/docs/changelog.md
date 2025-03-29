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

### Standard library changes

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

### Tooling changes

### ❌ Removed

- The `SIMD.roundeven()` method has been removed from the standard library.
  This functionality is now handled by the `round()` function.

- Error messages about the obsolete `borrowed` and `inout` keywords, as well as
  the obsolete `-> Int as name` syntax has been removed.

- The `StringableCollectionElement` trait has been removed in favor of
  `WritableCollectionElement`.

### 🛠️ Fixed
