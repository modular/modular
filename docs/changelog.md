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

- A new `IntervalTree` data structure has been added to the standard library.
  This is a tree data structure that allows for efficient range queries.

- `StringSlice` now supports several additional methods moved from `String`.
  The existing `String` methods have been updated to instead call the
  corresponding new `StringSlice` methods:

  - `split()`
  - `lower()`
  - `upper()`
  - `is_ascii_digit()`
  - `isupper()`
  - `islower()`
  - `is_ascii_printable()`
  - `rjust()`
  - `ljust()`
  - `center()`

- Added a `StringSlice.is_codepoint_boundary()` method for querying if a given
  byte index is a boundary between encoded UTF-8 codepoints.

### GPU changes

- `ctx.enqueue_function(compiled_func, ...)` is deprecated:

```mojo
  from gpu import thread_idx
  from gpu.host import DeviceContext

  fn func():
      print("Hello from GPU thread:", thread_idx.x)

  with DeviceContext() as ctx:
      var compiled_func = ctx.compile_function[func]()
      ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=4)
```

You should now pass the function directly to
`DeviceContext.enqueue_function[func](...)`:

```mojo
  with DeviceContext() as ctx:
      ctx.enqueue_function[func](grid_dim=1, block_dim=4)
```

### Tooling changes

#### Mojo Compiler

Mojo compiler now warns about parameter for with large loop unrolling factor
(>1024 by default) which can lead to long compilation time and large generated
code size. Set `--loop-unrolling-warn-threshold` to change default value to
a different threshold or to `0` to disable the warning.

### ❌ Removed

### 🛠️ Fixed
