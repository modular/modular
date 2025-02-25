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

- The `Buffer` struct has been removed in favor of `Span` and `NDBuffer`.

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

- `StringSlice.__getitem__(Slice)` will now raise an error if the provided slice
  start and end positions do not fall on a valid codepoint boundary. This
  prevents construction of malformed `StringSlice` values, which could lead to
  memory unsafety or undefined behavior. For example, given a string containing
  multi-byte encoded data, like:

  ```mojo
  var str_slice = "Hi👋!"
  ```

  and whose in-memory and decoded data looks like:

  ```text
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃          Hi👋!          ┃ String
  ┣━━┳━━━┳━━━━━━━━━━━━━━━┳━━┫
  ┃H ┃ i ┃       👋      ┃! ┃ Codepoint Characters
  ┣━━╋━━━╋━━━━━━━━━━━━━━━╋━━┫
  ┃72┃105┃    128075     ┃33┃ Codepoints
  ┣━━╋━━━╋━━━┳━━━┳━━━┳━━━╋━━┫
  ┃72┃105┃240┃159┃145┃139┃33┃ Bytes
  ┗━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━┛
   0   1   2   3   4   5   6
  ```

  attempting to slice bytes `[3-5)` with `str_slice[3:5]` would previously
  erroenously produce a malformed `StringSlice` as output that did not correctly
  decode to anything:

  ```text
  ┏━━━━━━━┓
  ┃  ???  ┃
  ┣━━━━━━━┫
  ┃  ???  ┃
  ┣━━━━━━━┫
  ┃  ???  ┃
  ┣━━━┳━━━┫
  ┃159┃145┃
  ┗━━━┻━━━┛
  ```

  The same statement will now raise an error informing the user their indices
  are invalid.

- Added an iterator to `LinkedList` ([PR #4005](https://github.com/modular/mojo/pull/4005))
  - `LinkedList.__iter__()` to create a forward iterator.
  - `LinkedList.__reversed__()` for a backward iterator.

  ```mojo
  var ll = LinkedList[Int](1, 2, 3)
  for element in ll:
    print(element[])
  ```

- The `round` function is now fixed to perform "round half to even" (also known
  as "bankers' rounding") instead of "round half away from zero".

- The `SIMD.roundeven()` method has been removed from the standard library.
  This functionality is now handled by the `round()` function.

- The `UnsafePointer.alloc()` method has changed to produce pointers with an
  empty `Origin` parameter, instead of with `MutableAnyOrigin`. This mitigates
  an issue with the any origin parameter extending the lifetime of unrelated
  local variables for this common method.

### GPU changes

- You can now skip compiling a GPU kernel first and then enqueueing it:

```mojo
  from gpu import thread_idx
  from gpu.host import DeviceContext

  fn func():
      print("Hello from GPU thread:", thread_idx.x)

  with DeviceContext() as ctx:
      var compiled_func = ctx.compile_function[func]()
      ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=4)
```

- You can now skip compiling a GPU kernel first before enqueueing it, and pass
a function directly to `ctx.enqueue_function[func](...)`:

```mojo
from gpu.host import DeviceContext

fn func():
    print("Hello from GPU")

with DeviceContext() as ctx:
    ctx.enqueue_function[func](grid_dim=1, block_dim=1)
```

However, if you're reusing the same function and parameters multiple times, this
incurs some overhead of around 50-500 nanoseconds per enqueue. So you can still
compile the function first and pass it to ctx.enqueue_function in this scenario:

```mojo
var compiled_func = ctx.compile_function[func]()
# Multiple kernel launches with the same function/parameters
ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
```

- The `shuffle` module has been rename to `warp` to better
  reflect its purpose. To uses now you will have to do

  ```mojo
  import gpu.warp as warp

  var val0 = warp.shuffle_down(x, offset)
  var val1 = warp.broadcast(x)
  ```

- `List.bytecount()` has been renamed to `List.byte_length()` for consistency
  with the String-like APIs.

### Tooling changes

#### Mojo Compiler

- Mojo compiler now warns about parameter for with large loop unrolling factor
  (>1024 by default) which can lead to long compilation time and large generated
  code size. Set `--loop-unrolling-warn-threshold` to change default value to
  a different threshold or to `0` to disable the warning.

- The Mojo compiler now only has one comptime interpreter.  It had two
  previously: one to handle a few cases that were important for dependent types
  (but which also had many limitations) in the parser, and the primary one that
  ran at "instantiation" time which is fully general. This was confusing and
  caused a wide range of bugs.  We've now removed the special case parse-time
  interpreter, replacing it with a more general solution for dependent types.
  This change should be invisible to most users, but should resolve a number of
  long-standing bugs and significantly simplifies the compiler implementation,
  allowing us to move faster.

### ❌ Removed

### 🛠️ Fixed
