# The `LLCL::AsyncValue` family of types

This document explains some of the concepts behind `AsyncValue` and related
types like `AsyncValueRef<T>`.

# `AsyncValue`

[`AsyncValue`](../include/LLCL/Runtime/AsyncValue.h) is conceptually similar to
[std::future](https://en.cppreference.com/w/cpp/thread/future), except that
`AsyncValue` does not let callers wait/block until the value becomes available.
Instead, the caller enqueues a closure that uses the value with
`AsyncValue::andThen`. `AsyncValue::emplace` will run any enqueued closures when
the value becomes available. This approach is similar to
[continuation passing](https://en.wikipedia.org/wiki/Continuation-passing_style).

Another major difference is that `AsyncValue` has built in support for error
handling: in addition to being completed by a future value, they may also be
completed by an error value.  All clients are expected to cope with (and
propagate) errors in a correct way.

`AsyncValue`s are heap allocated and reference counted.  You should use them
with the [`RCRef`](../include/LLCL/Support/RCRef.h) and
[`AsyncValueRef<T>`](../include/LLCL/Runtime/AsyncValueRef.h) classes whenever
possible to maintain their lifetime.

## Types and type erasure

An `AsyncValue` will eventually resolve to hold a value of some C++ type, but
this is dynamic and can happen after construction.  The `AsyncValue` type itself
is therefore type-erased: users can manipulate an `RCRef<AsyncValue>` without
knowing what type it will ultimately contain.  For example, you can enqueue a
closure with `AsyncValue::andThen()` without knowing the actual type that will
ultimately be contained in the `AsyncValue`. Type information is only needed
when *accessing* the contained data, for example with `AsyncValue::get<T>()` or
`AsyncValue::emplace<T>()`.

`RCRef<AsyncValue>` is used when working with a type-erased
`AsyncValue` and `AsyncValueRef<T>` is used when you know the element type `T`
that is stored in the `AsyncValue`.  It is preferable to use strong types if
you know them, but dynamic type-generic code sometimes doesn't.  
`AsyncValueRef<T>` implicitly converts to `RCRef<AsyncValue>`.

`AsyncValue` can hold any C++ type, including move-only and even non-movable
types, but all types need to be registered before use with
`AsyncValue::registerType<T>()`.  This registration logic is allows dense
storage of the payloads and data, and allows limited type reflection with the
`->isType<T>()` predicate.

## The states of `AsyncValue`

`AsyncValue` may be in four possible states: "unconstructed", "constructed",
"value available" and "error".  The final two states are considered to be
"ready" states - they happen when the future is resolved (either to a value or
an error) - all waiters are notified transitioning to a ready state, and you
cannot transition an `AsyncValue` back out of a ready state.

**"Unconstructed":** An `AsyncValue` in unconstructed state is obtained from the
`AsyncValue::createUnconstructed<T>` or `AsyncValueRef<T>::createUnconstructed`
static method.  In this state, any `andThen` requests are queued up until the
value transitions into a ready state.

**"Constructed":** An `AsyncValue` in constructed state is obtained from
`AsyncValue::createConstructed<T>` or `AsyncValueRef<T>::createConstructed`
static method, which take the arguments to the constructor.  This state is used
by code that finds it convenient to construct a C++ type directly into an
AsyncValue, manipulate it in place for a while, and eventually complete it by
transitioning to a ready state.  This can be useful for types that are
non-movable.  If transitioning to "value available", the waiter list is
notified.  When transitioning to "error", the value is destructed, the error is
installed and then the waiter list is notified.

**"Value Available":** This is the state that most `AsyncValue`s achieve where
they hold a completed C++ value and where all `andThen` waiters are notified.
You can directly create an `AsyncValue` in this state with
`AsyncValue::createReady<T>` or `AsyncValueRef<T>::createReady`,
but most cases will create one in unconstructed and transition to this state
with the `emplace(...)` method, or start in constructed state and transition
 with the `markReady()` method.

**"Error":** This state indicates that the computation creating the value had
an error.  You may create an `AsyncValue` directly in this state with the
`createError` method, but a more typical usage is to determine that an 
unconstructed or constructed `AsyncValue` had a problem, and transition it to
this state with the `setToError` method.

### Indirect Async Values

Beyond these four core states, you may run into a situation where you need to
create an `AsyncValue` before knowing what C++ type it will contain.  In this
case, you can create a special "IndirectAsyncValue" with the
`AsyncValue::createIndirect`, and resolve it with `resolveIndirect` method.  As
the name implies, this adds a level of indirection that allows you to create an
AsyncValue, and then fulfill it with another AsyncValue of concrete type later.  

For example, you might have some type generic code that resolves the type
depending on the input types:

```C++
// This works with both integer and string values forming "x+x" or "concat(x,x)"
// depending on what the argument resolves to.
RCRef<AsyncValue> genericAsyncDouble(RCRef<AsyncValue> input) {
  // Must create this value before knowing what type `input` is.
  RCRef<AsyncValue> result = AsyncValue::createIndirect(input->getRuntime());

  auto *inputPtr = input.getPointer(); // Watch out for order of evaluation.
  inputPtr->andThen([input = std::move(input), result = result.copy()]() {
    RCRef<AsyncValue> newVal;
    if (input.isType<int32_t>())
      newVal = AsyncValue::createReady<T>(input.get<int32_t>()*2);
    else {
      assert(input.isType<std::string>() && "unexpected type");
      const std::string &str = input.get<std::string>();
      newVal = AsyncValue::createReady<T>(str+str);
    }
    result->resolveIndirect(std::move(newVal));
  });

  return result;
}
```

The order of evaluation issue is a bit annoying.  If this were written as 
`input->andThen([input = std::move(input), ...` then we would have a problem.
C++ compilers before C++'20 do not specify the order of evaluation, and some
will do the move of 'input' into the capture list before reading from the
receiver.  FIXME: Address this with [Issue #35](https://github.com/modularml/modular/issues/35).

