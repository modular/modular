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

- References to aliases in struct types with unbound (or partially) bound
  parameters sets are now allowed as long as the referenced alias doesn't
  depend on any unbound parameters:

  ```mojo
  struct StructWithParam[a: Int, b: Int]:
    alias a1 = 42
    alias a2 = a+1

  fn test():
    _ = StructWithParams.a1 # ok
    _ = StructWithParams[1].a2 # ok
    _ = StructWithParams.a2 # error, 'a' is unbound.
  ```

### Standard library changes

- The design of the `IntLiteral` and `FloatLiteral` types has been changed to
  maintain their compile-time-only value as a parameter instead of a stored
  field. This correctly models that infinite precision literals are not
  representable at runtime, and eliminates a number of bugs hit in corner cases.
  This is made possible by enhanced dependent type support in the compiler.

- The `Buffer` struct has been removed in favor of `Span` and `NDBuffer`.

- The `InlineArray(unsafe_uninitialized=True)` constructor is now spelled `InlineArray(uninitialized=True)`.

- `Optional`, `Span`, and `InlineArray` have been added to the prelude.  You
   now no longer need to explicitly import these types to use them in your program.

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

- Added an iterator to `LinkedList` ([PR #4005](https://github.com/modular/max/pull/4005))
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

- The `SIMD` type now exposes 128-bit and 256-bit element types, with
  `DType.uint128`, `DType.int128`, `DType.uint256`, and `DType.int256`. Note
  that this exposes capabilities (and limitations) of LLVM, which may not always
  provide high performance for these types and may have missing operations like
  divide, remainder, etc.

- Several more packages are now documented.
  - `gpu` package - some modules in `gpu.host` subpackage are still in progress.
  - `compile` package
  - `layout` package is underway, beginning with core types, functions, and traits.

- A new `sys.is_compile_time` function is added. This enables one to query
whether code is being executed at compile time or not. For example:

```mojo
from sys import is_compile_time

fn check_compile_time() -> String:
   if is_compile_time():
      return "compile time"
   else:
      return "runtime"

def main():
    alias var0 = check_compile_time()
    var var1 = check_compile_time()
    print("var0 is evaluated at ", var0, " , while var1 is evaluated at ", var1)
```

will print `var0 is evaluated at compile time, while var1 is evaluated at runtime`.

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

- The methods on `DeviceContext`:

  - enqueue_copy_to_device
  - enqueue_copy_from_device
  - enqueue_copy_device_to_device

  Have been combined to single overloaded `enqueue_copy` method, and:

  - copy_to_device_sync
  - copy_from_device_sync
  - copy_device_to_device_sync

  Have been combined into an overloaded `copy` method, so you don't have
  to figure out which method to call based on the arguments you're passing.

- The `shuffle` module has been rename to `warp` to better
  reflect its purpose. To uses now you will have to do

  ```mojo
  import gpu.warp as warp

  var val0 = warp.shuffle_down(x, offset)
  var val1 = warp.broadcast(x)
  ```

- `List.bytecount()` has been renamed to `List.byte_length()` for consistency
  with the String-like APIs.

- The `logger` package is now documented.

- Large bigwidth integers are introduced. Specifically, the Int128, UInt128,
  Int256, and UInt256 are now supported.

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

- Direct access to `List.size` has been removed. Use the public API instead.

  Examples:

  Extending a List:

  ```mojo
  base_data = List[Byte](1, 2, 3)

  data_list = List[Byte](4, 5, 6)
  ext_data_list = base_data.copy()
  ext_data_list.extend(data_list) # [1, 2, 3, 4, 5, 6]

  data_span = Span(List[Byte](4, 5, 6))
  ext_data_span = base_data.copy()
  ext_data_span.extend(data_span) # [1, 2, 3, 4, 5, 6]

  data_vec = SIMD[DType.uint8, 4](4, 5, 6, 7)
  ext_data_vec_full = base_data.copy()
  ext_data_vec_full.extend(data_vec) # [1, 2, 3, 4, 5, 6, 7]

  ext_data_vec_partial = base_data.copy()
  ext_data_vec_partial.extend(data_vec, count=3) # [1, 2, 3, 4, 5, 6]
  ```

  Slicing and extending a list efficiently:

  ```mojo
  base_data = List[Byte](1, 2, 3, 4, 5, 6)
  n4_n5 = Span(base_data)[3:5]
  extra_data = Span(List[Byte](8, 10))
  end_result = List[Byte](capacity=len(n4_n5) + len(extra_data))
  end_result.extend(n4_n5)
  end_result.extend(extra_data) # [4, 5, 8, 10]
  ```

- Use of legacy argument conventions like `inout` and the use of `as` in named
  results now produces an error message instead of a warning.

- The `InlinedFixedVector` collection has been removed.  Instead, use
  `InlineArray` when the upper bound is known at compile time.  If the upper
  bound is not known until runtime, use `List` with the `capacity` constructor
  to minimize allocations.

- The `InlineList` type has been removed.  Replace uses with `List` and the
  capacity constructor, or an `InlineArray`.

### 🛠️ Fixed
