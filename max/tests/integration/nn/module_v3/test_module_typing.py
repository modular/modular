# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Static-typing regression tests for `Module.compile` / `CompiledModel`.

Locks in the MXF-442 contract: `Module[_P, _R].compile(...)` returns a
`CompiledModel[_P, _R]`, and `compiled(...)` is statically `_R` (not `Any`).

The bodies actually compile and run -- mirroring the pattern in
`test_module.py::test_compile` -- but the regression signal lives in the
`typing_extensions.assert_type` calls and the `# type: ignore[call-arg]`
sentinel below. Combined with `warn_unused_ignores = true` in
`pyproject.toml`, any regression that collapses `__call__` back to
`(*args: Any) -> Any` will mark the sentinel ignore as unused and break
the mypy aspect on this file.
"""

from __future__ import annotations

from max.experimental import random
from max.experimental.nn.module import (
    CompiledModel,
    Module,
    module_dataclass,
)
from max.experimental.tensor import Tensor, TensorType, defaults
from typing_extensions import assert_type


@module_dataclass
class UnaryTypingModule(Module[[Tensor], Tensor]):
    """Single Tensor in, single Tensor out -- the unary case."""

    bias: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x + self.bias


@module_dataclass
class MultiOutTypingModule(
    Module[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]
):
    """Two Tensors in, three-Tensor tuple out -- multi-output case."""

    bias: Tensor

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return a + self.bias, b + self.bias, a + b


def _input_type() -> TensorType:
    dtype, device = defaults()
    return TensorType(dtype, ["batch", "n"], device=device)


def test_compile_preserves_unary_p_r() -> None:
    module = UnaryTypingModule(bias=Tensor(0))
    compiled = module.compile(_input_type())
    assert_type(compiled, CompiledModel[[Tensor], Tensor])

    x = random.uniform([3, 3])
    result = compiled(x)
    assert_type(result, Tensor)


def test_compile_preserves_multi_output_p_r() -> None:
    module = MultiOutTypingModule(bias=Tensor(0))
    compiled = module.compile(_input_type(), _input_type())
    assert_type(
        compiled,
        CompiledModel[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    )

    a = random.uniform([3, 3])
    b = random.uniform([3, 3])
    result = compiled(a, b)
    assert_type(result, tuple[Tensor, Tensor, Tensor])

    first, second, third = result
    assert_type(first, Tensor)
    assert_type(second, Tensor)
    assert_type(third, Tensor)


def test_compile_rejects_call_arity_drift() -> None:
    """A unary `CompiledModel` rejects extra positional args at type-check time.

    Sentinel: the `# type: ignore[call-arg]` below is the regression alarm.
    Today (post-MXF-442), `compiled_unary(x, x)` is a real `call-arg` error
    and the ignore is justified. If a future change collapses
    `CompiledModel.__call__` back to `(*args: Any)`, mypy will stop flagging
    the line, `warn_unused_ignores=true` will mark the ignore unused, and
    this file will fail to type-check.
    """
    module = UnaryTypingModule(bias=Tensor(0))
    compiled = module.compile(_input_type())

    x = random.uniform([3, 3])
    # Wrapped in a lambda so this expression is never evaluated at runtime
    # (only type-checked). Calling `compiled(x, x)` at runtime would raise
    # inside `flatten_input_buffers` with a slot-count mismatch.
    _too_many_args = lambda: compiled(x, x)  # type: ignore[call-arg]
    del _too_many_args
