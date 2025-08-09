# Bridging compile-time and runtime values

## Problem statement

Mojo's strength lies in its ability to leverage compile-time
information for performance optimization while maintaining the
flexibility of runtime execution. However, the current type
system forces developers to make an early decision: either
a value must be a compile-time parameter or a runtime
argument. This creates several challenges:

1. **Function overload explosion**: When multiple arguments could
  potentially be compile-time known, developers must write exponentially
  many overloads to cover all combinations.
2. **Struct design limitations**: Structs face similar challenges
  where attributes might be known at compile-time in some contexts
  but not others (e.g., `Span` with constant size,
  matrices with known dimensions).
3. **Missed optimization opportunities**: Performance-critical functions
  must "hope" for constant folding rather than having guarantees about
  compile-time optimizations.
4. **Code generation bloat**: Functions accepting multiple variants
  (e.g., for compression methods, export formats) must decide to either
  generate code for all variants and increase binary size (argument) or
  limit themselves to a single variant, reducing flexibility (parameter).
5. **Default values are not optimized**: Default values in functions are
  massively used, but despite being known at compile-time, are rarely
  optimized for performance. Doing this would fall back to problem 1
  where many overloads are needed, especially for functions with multiple
  default arguments.

## Proposed solution

We propose introducing `ParameterOrArg` and `Parameter` to
the standard library. `ParameterOrArg` is a type that can
seamlessly accept both compile-time parameters and runtime
arguments, allowing the compiler and developer to defer the
decision to the caller and optimize accordingly. This requires
no compiler changes, just standard library additions.

### Core design

```mojo
# Simplified type definition for ParameterOrArg
struct ParameterOrArg[
    T: Copyable & Movable, //, _comptime_value: Optional[T]
    ](Copyable, Movable):
    """
    A value that can be either a compile-time parameter or runtime argument.
    """
    
    # Indicates whether this value is known at compile-time
    alias is_parameter = Self._comptime_value is not None
    
    # Access compile-time value (compilation fails if not compile-time)
    alias comptime_value = Self._comptime_value.value()
    
    # Access runtime value (compilation fails if compile-time)
    fn runtime_value(self) -> T
    
    # Access value regardless of when it's known
    fn value(self) -> T
```

### Usage example

```mojo
fn flexible_memcpy(
    dst: UnsafePointer[UInt8],
    src: UnsafePointer[UInt8], 
    size: ParameterOrArg[T=Int]
):
    @parameter
    if size.is_parameter:
        # Compiler can unroll, vectorize, and optimize for known size
        alias known_size = size.comptime_value
        # ... optimized implementation for compile-time known size
    else:
        # Standard runtime implementation
        var runtime_size = size.runtime_value()
        # ... generic implementation

# Usage examples:
fn example_usage():
    var dst = UnsafePointer[UInt8].alloc(100)
    var src = UnsafePointer[UInt8].alloc(100)
    
    # Case 1: Compile-time known size - gets optimized implementation
    flexible_memcpy(dst, src, Parameter[64]())
    
    # Case 2: Runtime size - uses generic implementation
    # with many branches
    var dynamic_size: Int = get_buffer_size()
    flexible_memcpy(dst, src, dynamic_size)
    
    # Case 3: Small literal size
    # We "hope" the compiler optimizes this thanks to 
    # the @always_inline (it's what's currently done).
    flexible_memcpy(dst, src, 16)
```

## Use cases

### 1. Multiple potentially-constant arguments

Without `ParameterOrArg`, functions with multiple potentially-constant
arguments require exponential overloads:

```mojo
# Current approach - 8 overloads needed!
fn process_data[width: Int, height: Int, channels: Int](data: Buffer): ...
fn process_data[width: Int, height: Int](data: Buffer, channels: Int): ...
fn process_data[width: Int, channels: Int](data: Buffer, height: Int): ...
fn process_data[width: Int](data: Buffer, height: Int, channels: Int): ...
# ... and 4 more combinations

# With ParameterOrArg - single function
fn process_data(
    data: Buffer,
    width: ParameterOrArg[T=Int],
    height: ParameterOrArg[T=Int], 
    channels: ParameterOrArg[T=Int]
):
    @parameter
    if width.is_parameter and height.is_parameter:
        # Optimize for known dimensions
        alias w = width.comptime_value
        alias h = height.comptime_value
        # ... specialized implementation

# Usage examples:
fn example_usage():
    var buffer = Buffer()
    
    # Case 1: All dimensions known at compile-time - maximum optimization
    process_data(
        buffer, Parameter[1920](), Parameter[1080](), Parameter[3]()
    )
    
    # Case 2: Mixed compile-time and runtime - partial optimization
    var dynamic_channels = detect_channels()
    process_data(
        buffer, Parameter[1920](), Parameter[1080](), dynamic_channels
    )
    
    # Case 3: All runtime dimensions - generic implementation
    var w: Int = get_width()
    var h: Int = get_height() 
    var c: Int = get_channels()
    process_data(buffer, w, h, c)
    
    # Case 4: Some known dimensions help optimization
    process_data(buffer, Parameter[640](), h, Parameter[3]())
```

### 2. Guaranteed optimizations for performance-critical code

Replace "we hope the compiler will do constant folding" with
guarantees in performance-sensitive functions:

```mojo
@always_inline
fn fast_memcopy[T: AnyType](
    dst: UnsafePointer[T],
    src: UnsafePointer[T],
    count: ParameterOrArg[Int]
):
    @parameter
    if count.is_parameter:
        alias n = count.comptime_value
        @parameter
        if n <= 4:
            # Inline the entire copy
            @unroll
            for i in range(n):
                dst[i] = src[i]
        elif n % 16 == 0:
            # Use aligned SIMD operations
            # ...
    else:
        # Runtime implementation with many if-else branches
        # See the memcpy implementation in the standard library
        _memcpy_runtime(
            dst, src, count.runtime_value() * sizeof[T]()
        )

# Usage examples:
fn example_usage():
    var dst = UnsafePointer[Int32].alloc(100)
    var src = UnsafePointer[Int32].alloc(100)
    
    # Case 1: We avoid branches, it's guaranteed to be optimized
    fast_memcopy(dst, src, Parameter[4]())
    
    # Case 3: Runtime count - uses generic implementation
    # and runtime branching depending on the count
    var dynamic_count = calculate_count()
    fast_memcopy(dst, src, dynamic_count)
    
    # Case 4: We "hope" the compiler optimizes this
    # thanks to the @always_inline (current behavior, unchanged)
    fast_memcopy(dst, src, 8)
```

### 3. Reducing code generation for variant functions

Functions that can potentially bloat the binary size allow the user
to choose if they want all code paths or just a specific one.

The user is in control of the binary size vs flexibility trade-off.

```mojo
fn compress_data(
    data: Buffer,
    method: ParameterOrArg[CompressionMethod]
) -> Buffer:
    @parameter
    if method.is_parameter:
        # Only generate code for the specific method
        alias m = method.comptime_value
        @parameter
        if m == CompressionMethod.GZIP:
            return _gzip_compress(data)
        elif m == CompressionMethod.ZSTD:
            return _zstd_compress(data)
        # ...
    else:
        # Generate code for all compression methods
        # make the binary larger, but flexible
        return _compress_dynamic(data, method.runtime_value())

# Usage examples:
fn example_usage():
    var input_data = Buffer()
    
    # Case 1: Known compression method - only GZIP code is generated
    var gzip_result = compress_data(
        input_data, Parameter[CompressionMethod.GZIP]()
    )
    
    # Case 2: Known compression method - only ZSTD code is generated
    var zstd_result = compress_data(
        input_data, Parameter[CompressionMethod.ZSTD]()
    )
    
    # Case 3: Runtime method selection
    # Every compression method ends up in the binary, more flexible
    var dynamic_result = compress_data(input_data, CompressionMethod.GZIP)
    var dynamic_result = compress_data(input_data, CompressionMethod.ZSTD)
```

### 4. Transfer the compile-time information to structs for later optimizations

Enable zero-cost abstractions that adapt to available
compile-time information.
**Note that another good use case could be tensor dimensions.** While
the syntax can seem strange, it's not exposed to the struct caller,
and the compiler can also make this nicer in the future
by allowing `*_, **_` in the types of struct attributes.

```mojo
struct Span[T: AnyType, _size_param: Optional[Int], //]:
    var data: UnsafePointer[T]
    var size: ParameterOrArg[_size_param]
    
    fn __init__(
        out self, data: UnsafePointer[T], size: ParameterOrArg[_size_param]
    ):
        self.data = data
        self.size = size
    
    fn get[index: Int](self) -> T:
        @parameter
        if __type_of(self.size).is_parameter:
            # Bounds check can be compile-time validated
            constrained[index < __type_of(self.size).comptime_value]()
        else:
            # Runtime bounds check
            debug_assert(index < self.size.runtime_value())
        
        return self.data[index]
    
    fn __len__(self) -> Int:
        return self.size.value()

# Usage examples:
fn example_usage():
    var buffer = UnsafePointer[Int32].alloc(100)
    
    # Case 1: Known size - compile-time bounds checking
    var fixed_span = Span(buffer, Parameter[100]())
    var value1 = fixed_span.get[42]()  # Bounds check at compile-time
    
    # Case 2: Runtime size - runtime bounds checking
    var dynamic_size = get_buffer_size()
    var dynamic_span = Span(buffer, dynamic_size)
    var value2 = dynamic_span[42]  # Bounds check at runtime
    
    # Case 3: Using with algorithms that benefit from size knowledge
    var sorted_span = Span(buffer, Parameter[5]())
    # May use different algorithm for known size, with 0 runtime branching
    quick_sort(sorted_span)
```

### 5. Specialized path for default values

We can also make this beneficial even
for users that don't bother with using `Parameter[...]()`.
Since default values are often (always?) known at compile-time, we can
ensure that the function is heavily optimized for the most common case.
A good example is `List.pop()` where the default value
(popping the last element) could be "specialized".

**Warning**: This use case currently requires a quite verbose syntax that
we can hopefully improve in the future with the help of
the compiler and tooling team.

```mojo
struct List[T: CollectionElement]:
    var data: UnsafePointer[T]
    var size: Int
    var capacity: Int
    
    fn pop[_param_i: Optional[Int] = -1](
        inout self, i: ParameterOrArg[_param_i] = ParameterOrArg[_param_i]()
    ) -> T:
        @parameter
        if i.is_parameter:
            @parameter
            if i.comptime_value == -1:
                # Optimized path for the common case (pop last)
                # No bounds checking needed, 
                # just check non-empty & decrement size
                debug_assert(self.size > 0, "Cannot pop from empty list")
                self.size -= 1
                return (self.data + self.size).take_pointee()^
            else:
                ...
        else:
            # Runtime idx - full bounds checking and logic
            ...

# Usage examples:
fn example_usage():
    var list = List[Int]([1, 2, 3, 4, 5])
    
    # Case 1: Default case - optimized path, no bounds checks
    var last = list.pop()  # Uses Parameter[-1]()
    
    # Case 2: Known index - compile-time bounds validation too
    var first = list.pop(Parameter[0]())
    
    # Case 3: Runtime index - full bounds checking
    var idx = get_user_index()
    var element = list.pop(idx)

    # Case 4: Known at compile time but without using Parameter[...]()
    # The compiler will decide what to do.
    # (depends notably on the inlining and constant folding)
    var element = list.pop(2)
```

## Implementation considerations

### Compiler integration

Currently, no compiler changes are needed.
But the compiler team might be involved with those:

- If we notice that the compiler doesn't produce
  zero-cost transfer of structs of size 0,
  we might ask for help there since this proposal relies on that.

There could be some slight, optional, quality-of-life
improvements in the future, such as:

- The use case **5** might benefit from
  a an improve syntax and way of displaying the signature.
- Allowing `*_, **_` in the types of struct attributes
- Changing `@parameter` to allow `@parameter(cond=value.is_parameter)` to
  disable the decorator if the value is only known at runtime, reducing
  code duplication without losing predictability.

Those features are not strictly necessary for the
initial proposal, but could help make the API more ergonomic.

### Performance guarantees

- Zero overhead compared to current overloads, we
  end up with empty structs when values are known at compile-time.

## Alternative designs considered

**Language-Level feature**: Build this into the language
syntax. While powerful, a library solution is
less invasive and can evolve independently. We can always
bake it into the language later if needed.

**Auto-detection by the compiler**: Make the compiler transfer
the information `some_funct("hello")` that the
argument is known at compile-time. This would
require significant compiler changes and could
lead to ambiguity in some cases. It would always require inlining.
Furthermore, the caller would lose control as using parameters
is not always desirable, for example it might increase the binary
size if called multiple times with arguments known at compile-time.
Mojo was designed to have predictable performance, and this
would make it harder to reason about.

## Migration path

### Steps

1. Introduce `ParameterOrArg` as experimental API.
2. Gradually adopt in performance-critical standard
  library functions by adding an overload, while not
  exposing those publicly. Make sure the assembly stays the same as before.

### Functions/struct that could benefit from `ParameterOrArg`

**Accepting this proposal does not mean those changes will be made.**

- `Span` to have the size be known at compile-time or runtime.
- `memcpy()` and other memory functions to optimize for
  compile-time known sizes as a lot of branching is present in
  the current implementation.
- `vectorize()` since the count can be known at compile-time or runtime.
- `SIMD.shift_right/shift_left` since they currently only
  have a compile-time known count, but could benefit from runtime
  flexibility without complexifying the API.
- `List.pop()` where the default value could be
  "specialized", thus avoiding bounds checks for the most
  common case (popping the last element).

And likely many more!

## Conclusion

`ParameterOrArg` bridges the gap between Mojo's powerful compile-time
capabilities and runtime flexibility. It enables developers to write
single, efficient implementations that adapt to available compile-time
information, reducing code duplication while maximizing performance.
It also follows Mojo's philosophy of letting the users decide what
optimization they want and avoid situations where we have to "hope" for
the compiler to do the right thing.
