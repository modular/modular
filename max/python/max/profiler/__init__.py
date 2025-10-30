# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""MAX profiler python package.

There are 3 ways to profile currently: context manager, decorator, and manual stack.

@tracing and Tracer need to have the MODULAR_ENABLE_PROFILING = 1 env variable set.

`Tracer` as a context manager:

```python
with Tracer("foo", color="modular_purple"):
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

`Tracer` as a manual trace stack manager:

```python
tracer = Tracer("foo", color="modular_purple")
tracer.push("bar")
# ...
tracer.pop()

# or as a context manager:
with Tracer("foo", color="modular_purple") as tracer:
    # The parent span is named "foo".
    tracer.push("bar")
    # The sub-span is named "bar".
    tracer.pop()
```
"""

from max._core.profiler import is_profiling_enabled, set_gpu_profiling_state
from max.profiler.tracing import Tracer, traced

__all__ = [
    "Tracer",
    "is_profiling_enabled",
    "set_gpu_profiling_state",
    "traced",
]
