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

- The Mojo comptime interpreter can now handle many more LLVM intrinsics,
   including ones that return floating point values.  This allows functions
   like `round` to be constant folded when used in a comptime context.

### Standard library changes

- A new `IntervalTree` data structure has been added to the standard library.
  This is a tree data structure that allows for efficient range queries.

- The `Char` type has been renamed to `Codepoint`, to better capture its
  intended purpose of storing a single Unicode codepoint. Additionally, related
  method and type names have been updated as well, including:

  - `StringSlice.chars()` to `.codepoints()` (ditto for `String`)
  - `StringSlice.char_slices()` to `.codepoint_slices()` (ditto for `String`)
  - `CharsIter` to `CodepointsIter`
  - `unsafe_decode_utf8_char()` to `unsafe_decode_utf8_codepoint()`

  - Make the iterator type returned by the string `codepoint_slices()` methods
    public as `CodepointSliceIter`.

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

- Added an iterator to `LinkedList` ([PR #4005](https://github.com/modular/mojo/pull/4005))
  - `LinkedList.__iter__()` to create a forward iterator.
  - `LinkedList.__reversed__()` for a backward iterator.

  ```mojo
  var ll = LinkedList[Int](1, 2, 3)
  for element in ll:
    print(element[])
  ```

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
