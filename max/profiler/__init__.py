# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""MAX profiler python package.

There are 3 ways to profile currently: context manager, decorator, and manual stack.

@tracing and Tracer need to have the MODULAR_ENABLE_PROFILING = 1 env variable set.

`Trace` Context manager:

```python
with Trace("foo", color="blue"):
    # Run `bar()` inside the profiling span.
    bar()
# The profiling span ends when the context manager exits.
```

`@traced` Decorator:

```python
@traced(message="baz", color="red")
def foo() -> None:
    # The span is named "baz".
    pass

@traced
def bar() -> None:
    # The span is named "bar".
    pass
```

`Tracer` Manual Stack manager (which can also be used as a context manager):

Note: The `Tracer` object can also be used as a context manager the same way the
`Trace` context manager is used.

```python
tracer = Tracer("foo", color="blue")
tracer.push("bar")
# ...
tracer.pop()

with Tracer("foo", color="blue") as tracer:
    # The parent span is named "foo".
    tracer.push("bar")
    # The sub-span is named "bar".
    tracer.pop()
```

"""

from max._core.profiler import (
    Trace,
    is_profiling_enabled,
    set_gpu_profiling_state,
)
from max.profiler.tracing import Tracer, traced

__all__ = [
    "Trace",
    "Tracer",
    "is_profiling_enabled",
    "set_gpu_profiling_state",
    "traced",
]
