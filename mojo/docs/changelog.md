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

### Tooling changes

### ❌ Removed

### 🛠️ Fixed
