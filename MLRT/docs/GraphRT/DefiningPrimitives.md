# Defining Primitives in GraphRT

Primitives are defined as C++ functions and registered dynamically with the
MEF executor's primitive registry at runtime by their registered string name.
This allows GraphRT to be an easily hackable system that is modular, low
dependence (because the runtime doesn't depend on the primitives), and is easy
to extend and friendly for experimentation.

This document gives a general overview of how to implement your own primitives,
with pointers to examples in the source base.

## How primitives work

As mentioned above, primitives are implemented as C++ functions, and these
functions take a `PrimitiveFrame` as their operand. They are registered with the
[`PrimitiveRegistry` class](../../include/MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h)
which looks like this:

```c++
class PrimitiveRegistry final {
public:
..
  /// Primitive implementations use this signature, acting on the data provided
  /// in the argument frame, and returning results through it.
  using PrimitiveImplFn = void (*)(RunnablePrimitiveDescriptor &frame);

  /// Add an implementation of a primitive to the registry.  Consider using
  /// the ADD_PRIMITIVE macro to enable various pretty forms for this.
  void add(StringLiteral name, PrimitiveImplFn fn);
  ...
};
```

This allows the executor itself to be completely operation independent: it works
on the *structure* of the operations, but doesn't know "what they do". The MLIR
syntax for the primitive has a directly corresponding runtime behavior:

1. Each SSA value operand and result in MLIR is an
   [`AsyncValue`](../../../AsyncRT/docs/AsyncValue.md) operand or result to the
   primitive.
2. Each MLIR attribute turns into a runtime pointer to static data of a
   corresponding C++ type. For example an MLIR `4 : i32` attribute will turn
   into a pointer to 32-bits of data containing a 4 (in little endian format).
   **These are passed in [ASCIIbetical
   order](https://en.wikipedia.org/wiki/ASCII#Character_order) by attribute
   name, not in lexical order in the `.mlir` file.** The attribute names are not
   encoded by default.
3. Symbol references like `@foo` and region bodies `{ ... }` in MLIR
   turn into "invokable" references that may be invoked in the executor.

The [`RunnablePrimitiveDescriptor
class`](../../include/MLRT/GraphRT/MEFExecutor/PrimitiveDescriptor.h) is the C++
class the MEF executor uses to wrap up all the runtime representation needed to
invoke and get the results from a primitive. While fundamentally all primitive
implementations poke the `RunnablePrimitiveDescriptor` class to get their work
done, it is a very low-level and not-very-type-safe approach.

## Registering primitives with the `ADD_PRIMITIVE` macro

Because the `RunnablePrimitiveDescriptor` class is so low-level and
error-prone, GraphRT provides an `ADD_PRIMITIVE` macro to help - you
should effectively always use the `ADD_PRIMITIVE` macro to register
primitives instead of directly poking at `RunnablePrimitiveDescriptor`.

This macro uses template metaprogramming magic to generate accessors for
`RunnablePrimitiveDescriptor` based on the signature for a C++ function.
There are lots of examples in the
[`lib/Primitives` library](../../lib/GraphRT/Primitives/)
but this section gives an overview of the key concepts.

### Registering simple primitives with operands and results

To see this in action, consider some simple primitives that take and return
values as SSA operands, for example with an `.mlir` syntax like this:

```mlir
  %add = grt.add.si32 %x, %x
  %cond = grt.lessequal.si32 %add, %one
  %ch2 = grt.print.si32 %add, %ch1
```

The `ADD_PRIMITIVE` macro allows primitives for these to be implemented and
registered like this:

```c++
static int32_t add_i32(int32_t lhs, int32_t rhs) { return lhs + rhs; }
static bool lessequal_i32(int32_t lhs, int32_t rhs) { return lhs <= rhs; }
static Chain print_i32(int32_t value, Chain chain) {
  printf("int32 = %d\n", value);
  fflush(stdout);
  return chain;
}
// ...
void M::registerGraphRTIntegerPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "grt.add.si32", add_i32);
  ADD_PRIMITIVE(registry, "grt.lessequal.si32", lessequal_i32);
  ADD_PRIMITIVE(registry, "grt.print.si32", print_i32);
}
```

The `ADD_PRIMITIVE` macro uses static reflection over the signature of the
function to generate something like this for the `lessequal_i32` example:

```C++
  registry.add("grt.lessequal.si32", [](RunnablePrimitiveDescriptor &frame) {
    assert(frame.getNumArgs() == 2 && frame.getNumResults() == 1 &&
           frame.getNumAttrs() == 0 && frame.getNumInvokables() == 0);
    frame.emplaceResult<bool>(0,
       lessequal_i32(frame.getArg<int32_t>(0), frame.getArg<int32_t>(1)));
  });
```

This automates the checking logic, unwraps the `AsyncValue`'s for the arguments
and generates the result `AsyncValue` as needed.

There are rare cases where you might want to directly access the AsyncValue for
an argument (for example maybe you want to `std::move` the reference to
micro-optimize some reference counts. In this case, you can take an
`AsyncValueRef` or `AnyAsyncValueRef` directly:

```C++
static AsyncValueRef<int32_t> test_noop(AsyncValueRef<int32_t> input) {
  // Don't allocate a new result, don't do any ref counting at all.
  return std::move(input);
}
```

### Variadic operands/results, invokables, and magic references

In some cases you may want to capture "all the rest" of the operands or results
in an operation. For example, consider the `grt.invoke` operation, which is
used like this in a `.mlir` file:

```mlir
   %y = grt.invoke @some_thing(%false, %a, %b) : (i1, si32, si32) -> (si32)

   %a, %b = grt.invoke @other_thing(%c1) : (si32) -> (f32, i1)
```

You can implement this using the `RemainingArguments` and `RemainingResults`
classes, which capture all of them as an `ArrayRef` of `AsyncValue`'s. For
example, the fully variadic `grt.invoke` operation is defined like this:

```C++
static void invoke(RemainingArguments arguments, RemainingResults results,
                   Invokable *target, Runtime &runtime) {
  assert(target->getNumArgs() == arguments.size() && "argument count mismatch");
  assert(target->getNumResults() == results.size() && "result count mismatch");
  target->invoke(arguments.get(), results.get(), runtime);
}
```

These also mix naturally with normal arguments and results, but
`RemainingArguments` must be listed before other operands, and any declared
results bind before `RemainingResults`.

This example also shows how to bind to symbol references like `@some_thing`
and region bodies: these are declared as `Invokable *` and `ADD_PRIMITIVE`
resolves them to the related invokable in the MEF file.

Finally, note that we also took a `Runtime &`. `ADD_PRIMITIVE` has direct
support for passing in a `Runtime &` or even a
`RunnablePrimitiveDescriptor &` for advanced cases: just ask and you will
receive.

### Producing errors and multiple results in synchronous primitives

Primitives in MEF may return multiple results, and both AsyncRT/GraphRT support
full error propagation with `AsyncValue`s. `ADD_PRIMITIVE` makes this extremely
simple and direct for synchronous primitives (ones that compute their values on
the calling thread, as opposed to using `addTask` to do it in the background).
You can use `std::pair` and `std::tuple` in the C++ result type, and you may
use the `ErrorOr<T>` template in arbitrary ways (even mixing them):

```C++
static ErrorOr<int32_t> halve_unless_odd(int32_t value) {
  if (value & 1)
    return Error("odd numbers are distasteful");
  return value / 2;
}

static std::pair<ErrorOr<int32_t>, int32_t> divrem(int32_t lhs, int32_t rhs) {
  if (rhs == 0)
    return {Error("divide by zero"), 0};
  return {lhs/rhs, lhs%rhs};
}
```

This is mostly useful for simple primitives like this that don't involve
asynchronous processing on a `WorkQueue`. For asynchronous cases, you need more
advanced patterns.

### Asynchronous results and custom result handling

While directly returning a simple type like `int32_t` is useful for trivial
cases, many primitives use the work queue in the runtime to parallelize the
computation of a result. As such, the primitive has to return back to the
executor before the result is ready. This means it needs to explicitly work
with the `AsyncValue` for the result.

To make this simpler and reduce boilerplate, you can declare these with the
`Result<T>` class. For example, it is a bad idea to perform unbounded amounts
of work on the executor's thread, so you might to initialize a vector on a
background task like this:

```C++
static void
vec_fill_non_prime_async_vector(int32_t size,
                                Result<std::vector<int32_t>> result) {
  // Invariants are typically checked before spawning background work.
  if (!isPrime(size))
    return result.setToError("only prime length vectors supported");

  // Fill in the value on a background task.
  addTask(result->getRuntime(), [size, result = result.copy()]() {
    std::vector<int32_t> resultData;
    // ... do something to fill in result in the background ...

    // After we've populated it, mark it ready with the value we built.
    result.emplace(std::move(resultData));
  });
}
```

While machine learning frameworks typically want to report errors synchronously,
there are some rare cases where you only realize on the background task that
there is a problem (for example maybe you want to treat divide by zero as an
error, which is dependent on the element data). `AsyncValue`'s need location
information to propagate their errors, and we don't copy that around by default.
To get it into your closure, use the `copyWithLocation()` method on `Result`:

```C++
static void
vec_fill_non_prime_async_vector(int32_t size,
                                Result<std::vector<int32_t>> result) {
  // Notice the `copyWithLocation` call here.
  addTask(result->getRuntime(), [size, result = result.copyWithLocation()]() {
    std::vector<int32_t> resultData;
    // ... do something to fill in result in the background ...
    if (data dependent bad thing happened) {
      result.setToError("some data dependent bad thing!");
      return;
    }
    result.emplace(std::move(resultData));
  });
}
```

The `Result<T>` type is used when you know the static C++ type of a specific
result, but sometimes, the result type is dynamic depending on the input values
or input attributes. In this case, the `IndirectResult` class will initialize
the result with an indirect `AsyncValue` that can be resolved to an arbitrary
type later (see
[test primitives](../../lib/GraphRT/Primitives/TestPrimitives.cpp) for
examples).

There are rare cases where you might want even more control over the AsyncValue
for a result. In these cases, you can declare your primitive to return an
`AnyAsyncValueRef` (for an untyped result) or `AsyncValueRef<T>` explicitly:

```C++
static AsyncValueRef<std::string>
really_custom(int32_t size, Runtime &runtime) {
  auto result = AsyncValueRef<std::string>::createConstructed(runtime);
  /// ... really custom logic ...
  return result;
}
```

These cases should be exceedingly rare.

### Working with attributes

The MEF file format supports encoding a number of MLIR attributes (including
string attributes, standard integer and floating point values, arrays of
integers and floats, `DenseElementAttr`s, and `DictionaryAttr`s) into the MEF
file. At runtime, the MEF file is mapped into memory, so your primitive can
get a raw pointer to the data in question. For example, consider a few simple
cases:

```mlir
  %x = grt.constant.si32 42

  // Same thing in verbose syntax:
  %x2 = "grt.constant.si32"() {value = 42 : si32} : () -> ()

  %z = "add.ints"() { value = dense<[1, 2, 3]> : tensor<3xsi32> } : () -> ()

  "some.thing"() { a = 3275124123121 : si64, b = "string value" } : () -> ()
```

Attributes in MLIR files are always canonicalized to be in alphabetical order by
their key (that is, if you move the string value before the `si64` in the last
example, they will still be emitted as "a" and then "b", so the `i64` will be
first), and MEF files don't encode the attribute names ("a" and "b" in that last
example). To handle these, you can implement the primitives using the
`MEFAttr<>` C++ class, and access the attribute with `*` or `->`:

```c++
static int32_t constant_i32(MEFAttr<int32_t> value) {
  return *value;  // Turns the attribute value into an AsyncValue result.
}
static int32_t add_ints(MEFAttr<ArrayRef<int32_t>> value) {
  return std::accumulate(value->begin(), value->end(), 0);
}
static void some_thing(MEFAttr<int64_t> a, MEFAttr<StringRef> b) {
  llvm::outs() << *a << " == " << b->str();
}
```

When receiving attributes, you need to use a well-known C++ type that matches
the datatype in question, for example `uint32_t` for `ui32` attributes,
and `StringRef` or `ArrayRef` for strings and `dense` elements.

In addition to low-level constant data access, you can also use MLIR array and
dictionary attributes (list of name, value pairs) which are accessed with the
`MEFListAttr` and `MEFDictionaryAttr` types in primitives. These supports
iteration, and returns the elements as a dynamic attribute (so you can cast to
access them as a concrete type). Dictionaries support finding individual
elements. These are not as efficient as the other attributes because they are
accessed indirectly through iterators, and the dictionary stores the names of
the keys. Accessing elements is done in the following simple examples:

```c++
static int32_t findNamedElement(MEFDictionaryAttr dict) {
  std::optional<MEFAttr<int32_t>> value = dict.find<int32_t>("myKey");
  if (!value)
    // Handle the error gracefully

  return *value;
}

static void handleAllElements(MEFDictionaryAttr dict) {
  for (auto element : dict) {
    if (MEFAttr<StringRef> str = dyn_cast<StringRef>(element.getValue())) {
      // handle `str`
    } else {
      MEFAttr<int32_t> value = element.getValue<int32_t>();
      // handle `value`
    }
  }
}

static void handleAllListElements(MEFListAttr list) {
  for (auto element : list) {
    if (MEFAttr<StringRef> str = dyn_cast<StringRef>(element)) {
      // handle `str`
    }
  }
}
```

**Dynamic Attributes:**

Some primitives might want to accept dynamic attributes and use reflection over
them to figure out what they are. You can do this with the `MEFDynamicAttr`
type like this:

```c++
static void takes_any_attribute(MEFDynamicAttr value) {
  if (auto intVal = dyn_cast<int32_t>(value))
    ...
  else if (auto stringVal = dyn_cast<StringRef>(value))
    ...
  else if (auto arrayVal = dyn_cast<ArrayRef<int32_t>>(value))
    ...
}
```

## Guidelines for Implementing Primitives

Now that you know how to register and implement primitives, it is important to
understand the general design principles that go into implementing them well.

Let's start by reminding ourselves of the general GraphRT and AsyncRT execution
model: we want to have a stable pool of threads that are executing non-blocking
work, and discovering and dispatching that work quickly - so massively multicore
machines are always kept busy with "things to do".

As such, real GraphRT programs generally aren't composed of simple integer
operations like we use for our examples, it is generally made up of more
complicated operations that do real work, typically in the background on lots of
threads in the AsyncRT work queue. In machine learning workloads, there is
typically a significant amount of parallelism within an operation kernel
(intra-op), but also a significant amount of graph-level parallelism as well
(inter-op).

With these use-cases in mind, here are some things to think about.

### Minimize "synchronous" work, prefer "asynchronous" work

The primitive is invoked by the MEF executor when its arguments become ready,
and because it is arbitrary C++ logic, they can do pretty much anything. We
break the behavior of a primitive in two major categories:
"synchronous" work done by the primitive itself on the invoker's thread, and
"asynchronous" work done on background threads (managed by the WorkQueue).

The *synchronous* work done by the primitive itself is sensitive: it is often
invoked from the thread that is dispatching the primitives, so taking an undue
amount of time will prevent other primitives from getting initiated in a timely
manner. This can delay generating work for other cores on your massive
multi-core machine and can delay launching kernels on accelerators.

That said, it is great for the primitive to launch arbitrary amounts of
*asynchronous* work that consumes a bunch of threads, can invoke hierarchical
parallelism or other fancy things, etc. Large multicore machines want lots of
bites of computation to keep all those cores busy. There are a few things
to keep in mind for an efficient and well balanced system:

1. Most important: all computation needs to be thread safe, and should not
   block. This is core to the AsyncRT computation model. Do not use things like
   `llvm::outs()` because it is not thread safe.
2. Don't do synchronous work that can take unbounded time (for example it is
   dependent on the input sizes). Fork/join parallelism is an extremely common
   pattern, which induces patterns with a lot of single-threaded regions in
   between the very-parallel forked regions. These single-threaded regions
   easily end up dominating execution time due to
   [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law). As such,
   keep the synchronous work small.
3. Least important: there is a (very) small amount of cost to creating a task
   and doing asynchronous work, so doing extremely small computations (like
   adding two integers) on the executor thread does make sense.

If you want to get really fancy, you can take advantage of the fact that GraphRT
allows primitives to include arbitrarily dynamic logic. For example, you could
conceptually do things like this:

```C++
static void vec_fill_vector(int32_t size, Result<std::vector<int32_t>> result) {
  auto generateElements = [](size_t size) -> std::vector<int32_t> {
    std::vector<int32_t> elements(size);
    // ... algorithm to generate the elements ...
    return elements;
  };

  // If we only have a few elements, do it on the executor thread.
  if (size < 4) {
    result.emplace(generateElements(size));
    return;
  }

  // Otherwise, do it on a background task.
  addTask(result->getRuntime(), [size, result = result.copy()]() {
    result.emplace(generateElements(size));
  });
}
```

TODO: We should quantify this and develop a better sense for the tradeoff point.

### Correctly managing resource lifetimes

The executor will completely skip execution of primitives that get fed error
values. This is great for implicit and automatic propagation of error values,
but it means that you have to be careful that the `AsyncValue`s own (and thus
deallocate) the resources in the program when they are destroyed. For example
consider this MEF program:

```mlir
mef.invokable @bad_design(%ch0: !mef.chain) -> !mef.chain {
  %x = accel.allocate_thing : !accel.thing

  %ch1 = accel.process_thing %x, %ch0

  // A "deallocate" op is almost always wrong!
  %ch2 = accel.deallocate_thing %x, %ch1
  mef.output %ch2 : !mef.chain
}
```

This could be (incorrectly!) implemented with primitives like this:

```c++
static MyAcceleratorType *allocate_thing() {
  return new MyCoolAcceleratorThing(...);
}
static void deallocate_thing(MyAcceleratorType *coolThing) {
  delete coolThing;
}
```

The problem is that the "deallocate" call may get skipped ... for example,
the `accel.process_thing` primitive could have a problem and report an error
through its result chain. In that case, the executor will skip the execution
of the deallocation hook and you'll have a leak (possibly of accelerator memory
or other resources!).

The solution to this is to take advantage of the fact that `AsyncValue` can hold
completely arbitrary C++ types, including ones that are not-even-movable. For
example, you could use this instead:

```c++
static AsyncValueRef<MyAcceleratorType> allocate_thing() {
  return AsyncValueRef<MyAcceleratorType>::createReady(...);
}
```

The "MyAcceleratorType" instance will be created inline into the `AsyncValue`
saving an extraneous allocation and automatically deleting it when the last use
of the `AsyncValue` is done. Sometimes you might want to get more fancy, for
example with C++ subclasses, shared references to concurrent-safe types like
thread safe hash tables, etc. In those cases, you can use standard C++ smart
pointers, for example like shared or unique pointers:

```c++
static void allocate_thing(Result<std::unique_ptr<MyAcceleratorType>> result) {
  result.emplace(new MyCoolAcceleratorThing(...));
}
```

As such, be on the lookup for "deallocator" primitives. They are almost always
the wrong thing. TODO: Should we completely ban pointers (and things like
`ArrayRef`/`StringRef`) in arguments and results?

## Non-strict primitives

By default, the MEF executor only invokes a primitive implementation function
when all of the arguments (SSA values in MLIR syntax) are "ready"
`AsyncValue`'s. When they become ready, it checks to see if any of them are
errors. If so, the error value is automatically propagated to the result value
of the primitive (so primitives don't have to handle errors explicitly). This is
the right default for almost all primitives, but some (notably control flow)
want to do more custom processing. For these cases, GraphRT supports a
"non-strict" primitive.

### What are non-strict primitives?

The defining feature of a GraphRT "non-strict" primitive is that it is invoked
when *any* argument (SSA value) becomes ready, instead of when *all* arguments
become available. Consider the `grt.invoke` operation for example, which invokes
a invokable passing a bunch of arguments:

```mlir
mef.invokable @process(%a: si32, %b: si32) -> (si32, si32) {
  %x = "some_expensive_multithreaded_compute"(%a) : (si32) -> si32
  %y = "independent_expensive_multithreaded_compute"(%b) : (si32) -> si32
  mef.output %x, %y : si32, si32
}

mef.invokable @example() {
  %a_val = "some_unpredictable_length_op"() : () -> si32
  %b_val = "other_unpredictable_length_op"() : () -> si32

  %res_0, %res_1 = grt.invoke @process(%a_val, %b_val) : (si32, si32) -> (si32, si32)
  ...
}
```

Notice the structure of this program: the `@process` invokable takes two
arguments, but has two different computation patterns depending on each of them.
The `grt.invoke` operation is defined as non-strict, which allows it to kick off
execution of its body as soon as *either* `%a_val` or `%b_val` are ready,
allowing the expensive compute to start as soon as possible.

This is really important for control flow-like operations (for example
`grt.invoke`, `grt.if`, etc) because they can end up taking lots of operands,
those operands can all have different computational chains that produce them,
and we don't want to block execution of the body for all of them to be
available. Doing so would destroy composability of these control flow operators

- for example not inlining a grt.invoke could have devastating and
unpredictable performance effects.

### Implementing a non-strict primitive

The key thing to remember when implementing a non-strict primitive is that their
body gets invoked before the operands are "ready", and (as such) those operands
may get resolved to error values. This means that the implementation of a
non-strict primitive needs to handle this. Notably, this also means that all
arguments **must be declared as `AsyncValue`'s**.

For example, a binary operation should be implemented like this:

```C++
// It is silly to do this for add of course, but this shows how to handle this.
static void nonstrict_add_i32(AsyncValueRef<int32_t> lhs,
                              AsyncValueRef<int32_t> rhs,
                              Result<int32_t> result) {
  // This may get invoked before lhs/rhs are resolved, and either or both may
  // resolve to an error, so we need to handle that.

  // Invoke the lambda when both AsyncValue's resolve.
  andThenSync(
      [result = result.copy()](AsyncValueRef<int32_t> lhs,
                               AsyncValueRef<int32_t> rhs) {
        // Okay, both are "ready" but either may be an error.  We must
        // explicitly handle this.
        if (lhs->isError())
          result->setToError(lhs->takeDiagnostic());
        else if (rhs->isError())
          result->setToError(rhs->takeDiagnostic());
        else
          result.emplace(lhs.get() + rhs.get());
      },
      std::move(lhs), std::move(rhs));
}
```

There are lots of ways to handle the async operands of course, please take a
look at the primitives implemented in
[ControlFlowPrimitives.cpp](../../lib/GraphRT/Primitives/ControlFlowPrimitives.cpp)
for examples.

### Declaring a non-strict primitive in ODS

The next question is: how do you tell GraphRT that your primitive is non-strict?
The contract is quite different, and invoking a strict primitive too early would
cause crashes (because they'd unconditionally try to access `AsyncValue`'s that
are not ready). As such, you have to explicitly register the operation in
question with ODS (and the "mlir to mef" translator needs to know about the
operation). You then adopt the `NonStrictOp` trait.

For example:

```tablegen
def InvokeOp : GRTOp<"invoke", [NonStrictOp, ...]> {
def IfOp : GRTOp<"if", [NonStrictOp...]> {
```

The MLIR to MEF translator notices the presence of this trait, and encodes the
MEF entry for the primitive instances differently, allowing the executor to
efficiently dispatch the primitive when one operand becomes available.
