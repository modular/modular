---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

- Types can parameterize the `out` argument modifier when they want to be
  bindable to alternate address spaces, e.g.:

  ```mojo
  struct MemType(Movable):
    # Can be constructed into any address space.
    def __init__[addr_space: AddressSpace](out[addr_space] self):
        ...

    # Only constructable into GLOBAL address space.
    def __init__(arg: Int, out[AddressSpace.GLOBAL] self):
        ...
  ```

- Mojo now supports building types that support implicit conversions for
  widening origins, allowing code like this to "just work" without rebind:

  ```mojo
  def origin_superset_conversion(
    a: String, b: String, c: Bool
  ) -> Pointer[String, origin_of(a, b)]:
    if c:  # These pointers implicitly convert.
        return Pointer(to=a)
    else:
        return Pointer(to=b)
  ```

- Types may now be conditionally "ImplicitlyDestructible" with a where clause:

  ```mojo
  @explicit_destroy("Message when implicitly destroyed")
  struct ConditionallyLinearType[T: AnyType](
      ImplicitlyDestructible where conforms_to(T, ImplicitlyDestructible)
  ):
      var data: Self.T
  ```

## Language changes

- Support for "set-only" accessors has been removed. You need to define a
  `__getitem__` or `__getattr__` to use a type that defines the corresponding
  setter. This eliminates a class of bugs determining the effective element
  type.

- Implicit `std` imports are now an error, following a period of deprecation.
  Imports from the standard library must now be fully qualified. The compiler
  thus no longer squats on these module names, paving the way for user modules
  named `algorithm`, `memory`, etc.

- Specifying `ABI="C"` in an `@export` decorator is now deprecated; `abi("C")`
  should be used instead.

  ```mojo
  @export("old", ABI="C")
  def old(): pass

  @export("new")
  def new() abi("C"): pass
  ```

## Library changes

- The `ImplicitlyCopyable`, `Intable`, and `Equatable` traits no longer
  inherit from `ImplicitlyDestructible`. Generic code that relied on
  receiving the destructor bound transitively through these traits (or
  through `Comparable`, which inherits from `Equatable`) must now spell it
  out explicitly, for example
  `T: ImplicitlyCopyable & ImplicitlyDestructible`. In practice, most
  generic code should prefer `T: Copyable` instead, per the guidance in
  `ImplicitlyCopyable`'s docstring.

- Changed `Idx` to a `comptime` alias for `ComptimeInt`. Use `Idx[value]`
  instead of `Idx[value]()` for compile-time coordinates.

- Added `is_trivially_movable`, `is_trivially_copyable`, and
  `is_trivially_destructible` to `std.memory`. These helper functions
  return whether a type's move constructor, copy constructor, or destructor is
  trivial (i.e., a bit-copy or a no-op).

- Added `std.gpu.host.CompletionFlag`, a non-owning handle to an MLRT
  `M::Driver::CompletionFlag` (an 8-byte slot in pinned host memory mapped
  into a device's address space). Pairs with the new
  `DeviceStream.wait_for_host_value(flag, value)` method, which stalls the
  stream until the flag's 64-bit slot equals the given value. Corresponds to
  CUDA's `cuStreamWaitValue64` and captures cleanly into a CUDA graph as a
  wait-value node, letting a CPU thread (or an AsyncRT worker dispatched by
  `enqueue_host_func`) gate a GPU stream on host-produced data without a
  second stream or a blocking host-function callback. Currently CUDA-only;
  other backends raise.

- `Coord`, `coord()`, `Idx`, `ComptimeInt`, `RuntimeInt`, and related coordinate
  helpers now live in the standard library module
  [`std.utils.coord`](/docs/std/utils/coord/). The
  [`layout.coord`](/docs/layout/coord/) module re-exports the same symbols for
  layout and kernel code; `layout` also hoists the common names at package
  scope for convenience.

- Python -> Mojo FFI calls registered through `PythonModuleBuilder` and
  `PythonTypeBuilder` have significantly reduced per-call overhead:

  - Non-kwargs callables registered with `def_function` / `def_method` /
    `def_staticmethod` now use CPython's `METH_FASTCALL` calling
    convention rather than `METH_VARARGS`. Kwargs-accepting functions
    still use `METH_VARARGS | METH_KEYWORDS`.

  - `PythonObject.__del__` skips the `PyGILState_Ensure` /
    `PyGILState_Release` round-trip when the current thread already
    holds the GIL (checked via `PyGILState_Check`). On the common
    Python -> Mojo FFI path (where CPython hands the callee an
    already-held GIL) the destructor pays just the check and a direct
    `Py_DecRef`. The public contract is unchanged - dropping a
    `PythonObject` from a thread that does not hold the GIL remains
    safe.

  - `Int(py=obj)` and `Scalar[IntDType](py=obj)` fast-path exact
    Python `int` via `PyLong_AsSsize_t`.

- Added `TileTensor.copy_from()` and `TileTensor.split()` for copying between
  compatible tile views and splitting tiles into static or runtime-sized
  partitions.

- `String.as_bytes_mut()` has been renamed to `String.unsafe_as_bytes_mut()`, to
  reflect that writing invalid UTF-8 to the resulting `Span[Byte]` can lead to
  later issues like out of bounds access.

- A new `BinaryHeap` collection has been added to the `std.collections` module.
  This is a list-backed binary max-heap.

- The core collection types no longer require their element type to be
  `Copyable` — `Movable & ImplicitlyDestructible` is now the minimum bound.
  This applies to `List[T]`, `Deque[T]`, `LinkedList[T]`, `InlineArray[T,
  size]`, and the value type `V` of `Dict[K, V, H]` (and the underlying
  `SwissTable`/`SwissTableEntry` and `OwnedKwargsDict`).

- `reflect[T]` is now a `comptime` alias for the `Reflected[T]` handle type
  rather than a function returning a zero-sized handle instance. All methods on
  `Reflected[T]` are `@staticmethod`s, and the type is no longer constructible.
  Drop the parens at call sites:

  ```mojo
  # Before
  comptime r = reflect[Point]()
  print(r.field_count())
  print(reflect[Point]().name())
  comptime y_handle = reflect[Point]().field_type["y"]()
  var v: y_handle.T = 3.14

  # After
  comptime r = reflect[Point]
  print(r.field_count())
  print(reflect[Point].name())
  comptime y_handle = reflect[Point].field_type["y"]
  var v: y_handle.T = 3.14
  ```

  `field_type[name]` is now a parametric `comptime` member alias that yields
  `Reflected[FieldT]` directly — no trailing `()`, and the result is fully
  composable (e.g. `reflect[T].field_type["x"].name()`). The previously
  deprecated free functions `get_type_name`, `get_base_type_name`, and the
  `struct_field_*` family (along with the `ReflectedType[T]` wrapper) have been
  removed; use the corresponding methods on `reflect[T]`:

  | Removed                                 | Replacement                              |
  |-----------------------------------------|------------------------------------------|
  | `get_type_name[T]()`                    | `reflect[T].name()`                      |
  | `get_base_type_name[T]()`               | `reflect[T].base_name()`                 |
  | `is_struct_type[T]()`                   | `reflect[T].is_struct()`                 |
  | `struct_field_count[T]()`               | `reflect[T].field_count()`               |
  | `struct_field_names[T]()`               | `reflect[T].field_names()`               |
  | `struct_field_types[T]()`               | `reflect[T].field_types()`               |
  | `struct_field_index_by_name[T, name]()` | `reflect[T].field_index[name]()`         |
  | `struct_field_type_by_name[T, name]()`  | `reflect[T].field_type[name]`            |
  | `struct_field_ref[idx, T](s)`           | `reflect[T].field_ref[idx](s)`           |
  | `offset_of[T, name=name]()`             | `reflect[T].field_offset[name=name]()`   |
  | `offset_of[T, index=index]()`           | `reflect[T].field_offset[index=index]()` |
  | `ReflectedType[T]`                      | `Reflected[T]`                           |

- Added `ReflectedFn[func]`, a function-side reflection handle accessed via
  the `reflect_fn[func]` `comptime` alias. Exposes function introspection
  through static methods, paralleling the type-side `Reflected[T]` API:

  ```mojo
  from std.reflection import reflect_fn

  def my_func(x: Int) -> Int:
      return x + 1

  def main():
      print(reflect_fn[my_func].display_name())  # "my_func"
      print(reflect_fn[my_func].linkage_name())  # mangled symbol name
  ```

- Added `alloc`, `free`, and `Layout` in `memory.alloc` for layout-aware memory
  allocation. A `Layout[T]` bundles an element count and alignment into a
  single value that is passed to both `alloc` and `free`, keeping size and
  alignment requirements explicit and co-located at every call site.

  ```mojo
  from memory import alloc, free, Layout

  var layout = Layout[Int32](count=4)
  var ptr = alloc(layout)
  # ... initialize & use ptr ...
  free(ptr, layout)
  ```

- The default `seed` for `random.Random`, `random.NormalRandom`, and the
  internal `_PhiloxWrapper` has changed from `0` to `0x3D30F19CD101`
  (67280421310721) to match PyTorch's `at::Philox4_32_10` default. Calls
  that omitted the `seed` argument will now produce a different output
  stream; pass `seed=0` explicitly to keep the previous behavior.

- Added `nth()` as a default method on the `Iterator` trait. It advances the
  iterator by `n` elements (destroying them) and returns the next element, or
  `None` if the iterator runs out before reaching index `n`.

  ```mojo
  var l = [10, 20, 30, 40]
  print(iter(l).nth(0).value())   # 10
  print(iter(l).nth(3).value())   # 40
  var missing = iter(l).nth(10)   # None (Optional)
  ```

- `String` and `StringSlice` now have a keyword only `string[codepoint=...]`
  that indexes by unicode codepoint offsets.

- Several `StringSlice` constructors are now deprecated.

  - `StringSlice(ptr=..., length=...)` is deprecated; use
    `StringSlice(unsafe_from_utf8=Span(...))` instead.
  - `StringSlice(unsafe_from_utf8_ptr=...)` (taking a raw nul-terminated
    `UnsafePointer[Byte]` or `UnsafePointer[c_char]`) is deprecated; construct
    a `CStringSlice` from the pointer and use the new
    `StringSlice(unsafe_from_utf8=CStringSlice(...))` constructor instead.

- PythonObject convertibility got simplified and cleaned up. When working with
  types that required custom conversions to `PythonObject`, we used to write
  code like this:

  ```mojo
  struct MyCustomType(ConvertibleToPython, ImplicitlyCopyable):
     def to_python_object(var self) raises -> PythonObject:
        return PythonObject( ... custom logic ...)

  def hi_python(a: Some[ImplicitlyCopyable & ConvertibleToPython]) raises:
      print(t"Hi, {a.to_python_object()}!")

  def example():
      hi_python(MyCustomType())
  ```

  This approach allows custom types to implement `ConvertibleToPython` to get a
  domain specific encoding as a Python object. Mojo has simplified this by
  making all `ConvertibleToPython` types implicitly convert to `PythonObject`,
  so this can/should be simplified to:

  ```mojo
  def hi_python(a: PythonObject) raises:
      print(t"Hi, {a}!")
  ```

- The CPython FFI bindings now carry the `abi("C")` effect. User-written Python
  extension callbacks passed to `def_py_c_function`, `def_py_c_method`, or
  `PyCapsule_New` must add `abi("C")` to their signatures, e.g.
  `def my_func(self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:`.
  Functions registered through the higher-level `def_function`, `def_method`,
  and `def_staticmethod` paths are unaffected.

- Added `take()` and `drop()` iterator adapters to `std.itertools`.
  `take(iter, n)` yields the first `n` elements, and
  `drop(iter, n)` drops the first `n` elements. They compose
  naturally to select sub-ranges of any iterable:

  ```mojo
  from std.itertools import take, drop

  var nums = [1, 2, 3, 4, 5]
  for x in take(drop(nums, 1), 3):
      print(x)  # 2, 3, 4
  ```

- The `Indexer` trait no longer inherits from `ImplicitlyDestructible`.
  Generic code that relied on receiving the destructor bound transitively
  through this trait must now spell it out explicitly, for example
  `T: Indexer & ImplicitlyDestructible`.

## Tooling changes

- The `mojo` compiler will now print the filename and line number in diagnostics
  that point to inaccessible source locations (e.g., from precompiled libraries)
  instead of a location at the top of the main file:

  ```text
  # Before
  $> mojo example.mojo
  /path/to/example.mojo:33:16: error: invalid call to '__setitem__': violated constraint
     vec[base + i] = values[i].cast[dtype]()
     ~~~^~~~~~~~~~

  /path/to/example.mojo:1:1: note: constraint declared here evaluated to False, expected 'mut'
  from std.algorithm.functional import elementwise
  ^
  /path/to/example.mojo:1:1: note: function declared here
  from std.algorithm.functional import elementwise
  ^

  # After
  $> mojo example.mojo
  /path/to/example.mojo:33:16: error: invalid call to '__setitem__': violated constraint
      vec[base + i] = values[i].cast[dtype]()
      ~~~^~~~~~~~~~

  max/kernels/src/layout/layout_tensor.mojo:2092: note: constraint declared here evaluated to False, expected 'mut'
  max/kernels/src/layout/layout_tensor.mojo:2090: note: function declared here
  ```

- The `mojo` compiler now provides more useful diagnostics in the case that
  source information is unavailable by synthesizing a declaration and
  pretty-printing it.

  For example, instead of the following, with no contextual information after
  the 'here':

  ```text
  /path/to/file.mojo:2092: note: function declared here:
  ```

  The user will now see:

  ```text
  /path/to/file.mojo:2092: note: function declared here:
  __setitem__[*Tys: Indexer](self, *args: *Tys.values, *, val: SIMD[dtype, Self.element_size]) where mut
  ```

  The coverage and quality of diagnostics in such cases will continue to improve
  in subsequent releases.

- The `mojo package` command has renamed to `mojo precompile`. Similarly, the
  `.mojopkg` file extension has been deprecated; favor the `.mojoc` file
  extension instead.

  ```text
  # Before
  mojo package my_package -o my_package.mojopkg

  # After
  mojo precompile my_package -o my_package.mojoc
  ```

- Added `mojo --print-cache-location` and `mojo --clear-cache` for inspecting
  and clearing the on-disk Mojo compile cache (`.mojo_cache`). The resolved
  path honors the existing precedence (`MODULAR_CACHE_DIR`, `MODULAR_HOME`,
  `MODULAR_DERIVED_PATH`, `XDG_CACHE_HOME`, etc.). `--clear-cache` prompts for
  confirmation by default; pass `-f` (or `--force`) to skip the prompt for
  scripting use.

  ```text
  $ mojo --print-cache-location
  /home/you/.cache/modular/.mojo_cache

  $ mojo --clear-cache
  This will remove the Mojo compile cache at:
    /home/you/.cache/modular/.mojo_cache
  Proceed? [y/N] y
  Removed /home/you/.cache/modular/.mojo_cache

  $ mojo --clear-cache -f   # no prompt
  ```

## GPU programming

- Added `DeviceContextList[size]` in `std.gpu.host`: a fixed-size,
  `Copyable`/`ImplicitlyCopyable`/`Sized` collection of `DeviceContext` values.
  Multi-device custom-op `execute` methods now receive a `DeviceContextList[N]`
  — the graph compiler synthesizes one from the per-device contexts attached to
  the op via a variadic constructor. Kernels can index into it with
  `dev_ctxs[i]` (runtime) or `dev_ctxs.__getitem_param__[i]()` (comptime), and
  iterate with `len()`. This replaces the previous `DeviceContextPtrList`
  pattern.

  ```mojo
  from gpu.host import DeviceContext, DeviceContextList

  @compiler.register("mo.distributed.allreduce.sum")
  struct DistributedAllReduceSum:
      @staticmethod
      def execute[
          dtype: DType, rank: Int, target: StaticString, _trace_name: StaticString,
      ](
          outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
          inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
          signal_buffers: MutableInputVariadicTensors[dtype=DType.uint8, rank=1, ...],
          dev_ctxs: DeviceContextList,
      ) capturing raises:
          comptime num_devices = inputs.size
          # ... use dev_ctxs[i] per device ...
  ```

- `DeviceContext.enqueue_function[func]` and
  `DeviceContext.compile_function[func]` now accept a single kernel argument
  instead of requiring it to be passed twice. The previous two-argument forms
  `enqueue_function[func, func]` and `compile_function[func, func]` are
  deprecated. The transitional `enqueue_function_experimental` and
  `compile_function_experimental` aliases are also deprecated; switch to
  `enqueue_function` / `compile_function`.

  ```mojo
  # Before
  ctx.enqueue_function[my_kernel, my_kernel](grid_dim=1, block_dim=1)
  ctx.enqueue_function_experimental[my_kernel](grid_dim=1, block_dim=1)

  # After
  ctx.enqueue_function[my_kernel](grid_dim=1, block_dim=1)
  ```

## Removed

- The `DeviceContextPtr` and `DeviceContextPtrList` types have been removed
  from `std.runtime.asyncrt`. Custom-op `execute` methods now take
  `DeviceContext` directly (or `Optional[DeviceContext]` where the context is
  genuinely optional), and multi-device ops take `DeviceContextList[N]` (see
  the new entry under *Library changes*). The helpers `get_device_context()`
  and `get_optional_device_context()` are no longer needed — pass the
  `DeviceContext` through directly. The `CpuDeviceContext` runtime always
  supplies a real context for the CPU path, so the nullable wrapper is no
  longer required.

  ```mojo
  # Before
  from runtime.asyncrt import DeviceContextPtr, DeviceContextPtrList

  @compiler.register("my_op")
  struct MyOp:
      @staticmethod
      def execute[target: StaticString](
          output: OutputTensor,
          input: InputTensor,
          ctx: DeviceContextPtr,
      ) raises:
          var gpu_ctx = ctx.get_device_context()
          ...

  # After
  from gpu.host import DeviceContext

  @compiler.register("my_op")
  struct MyOp:
      @staticmethod
      def execute[target: StaticString](
          output: OutputTensor,
          input: InputTensor,
          ctx: DeviceContext,
      ) raises:
          ...
  ```

- The `use_blocking_impl` parameter has been removed from `elementwise` (in
  `std.algorithm.functional`), and the analogous
  `single_thread_blocking_override` parameter has been removed from the
  reduction APIs (`reduce`, `max`, `min`, `sum`, `product`, `mean` in
  `std.algorithm.reduction`). These operations now always dispatch work the same
  way, with a single worker used automatically when the problem size is small,
  so the blocking variants are no longer needed.

- The legacy `fn` keyword now produces an error instead of a warning. Please
  move to `def`.

- The previously-deprecated `constrained[cond, msg]()` function has been
  removed. Use `comptime assert cond, msg` instead.

- The previously-deprecated `Int`-returning overload of `normalize_index` has
  been removed. Use the `UInt`-returning overload (or write the index
  arithmetic inline, e.g. `x[len(x) - 1]`).

- The previously-deprecated default `UnsafePointer()` null constructor has
  been removed. To model a nullable pointer use
  `Optional[UnsafePointer[...]]`. For a non-null placeholder for delayed
  initialization, use `UnsafePointer.unsafe_dangling()`.

- The deprecated free-function reflection API in `std.reflection` has been
  removed. Use the unified `reflect[T]() -> Reflected[T]` API instead.

  Migration table:

  - `struct_field_count[T]()` → `reflect[T]().field_count()`
  - `struct_field_names[T]()` → `reflect[T]().field_names()`
  - `struct_field_types[T]()` → `reflect[T]().field_types()`
  - `struct_field_index_by_name[T, name]()` →
    `reflect[T]().field_index[name]()`
  - `struct_field_type_by_name[T, name]()` →
    `reflect[T]().field_type[name]()`
  - `struct_field_ref[idx](s)` → `reflect[T]().field_ref[idx](s)`
  - `is_struct_type[T]()` → `reflect[T]().is_struct()`
  - `offset_of[T, name=...]()` → `reflect[T]().field_offset[name=...]()`
  - `offset_of[T, index=...]()` → `reflect[T]().field_offset[index=...]()`
  - `ReflectedType[T]` → `Reflected[T]`

- `String` and `StringSlice` can now be sliced by codepoints, e.g.
  `String("🔄🔥🔄")[codepoint=1:2]` returns `"🔥"`.

- `String` and `StringSlice` can now be indexed by graphemes, e.g.
  `String("👨‍🚀🧑‍🌾क्षि")[grapheme=1]` returns `"🧑‍🌾"`.

## Fixed

- Reduced the virtual address space reserved by every `mojo` invocation by
  ~1 GiB. The JIT memory mapper's reservation granularity was 1 GiB, so each
  fresh reservation was rounded up to that size and mmapped
  `PROT_READ|PROT_WRITE`, inflating `VmPeak` and counting against Linux
  `RLIMIT_AS`. This caused non-deterministic OOM crashes in
  `libKGENCompilerRTShared.so` when two `mojo` processes ran concurrently on
  memory-constrained CI runners (e.g. GitHub Actions free-tier, 7 GiB). The
  granularity is now 64 MiB; large compiles still work because the mapper
  reserves additional slabs on demand.
  ([Issue #6433](https://github.com/modular/modular/issues/6433))

- Attempting to import a source Mojo package from a broken symlink will no
  longer result in a compiler crash.
  ([Issue #6424](https://github.com/modular/modular/issues/6424))

- `MODULAR_NVPTX_COMPILER_PATH` is now part of mojo cache location so that when
  switching to a different `ptxas` CUBIN cache will not hit those were
  generated before the switch.
  ([Issue #6540](https://github.com/modular/modular/issues/6549))

- Fixed the `mojo` compiler incorrectly emitting AVX-512 instructions on
  hosts where the CPU model (e.g. `znver4`) advertises AVX-512 but the OS
  has not enabled it in XCR0 — for example, inside Docker containers on
  GitHub Actions. Host CPU features are now cross-checked against the
  runtime CPUID view, so features the kernel withholds no longer cause
  `SIGILL` at runtime.
  ([Issue #6413](https://github.com/modular/modular/issues/6413))
