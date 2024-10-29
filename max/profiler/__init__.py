# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""MAX profiler python package.

There are 2 ways to profile currently: context manager or decorator.
(Decorators need to have the MODULAR_ENABLE_PROFILING = 1 env variable set.)

Context manager:

```python
with Trace("foo", color="blue"):
    # Run `bar()` inside the profiling span.
    bar()
# The profiling span ends when the context manager exits.
```

Decorator:

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
"""

from max._profiler import Trace
from max.profiler.tracing import traced

__all__ = [
    "Trace",
    "traced",
]
