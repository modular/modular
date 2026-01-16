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

from collections.abc import Callable, Iterable
from typing import Any

from max.driver import CPU, Accelerator
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.nn.module_v3 import Module


class CompileWrapper:
    def __init__(
        self,
        compile_target: Callable | Module,
        input_types: Iterable[TensorType] | None = None,
    ) -> None:
        """Initialize the CompileWrapper.

        Args:
            compile_target: The function or module to be compiled.
            input_types: A list of input types (TensorTypes) required for compilation.

        Raises:
            ValueError: If input_types is not provided.
        """
        if input_types is None:
            raise ValueError(
                f"input_types must be provided for compilation of {compile_target.__name__}."
            )

        self.is_module = False
        if isinstance(compile_target, Module):
            self.is_module = True
            self.session = compile_target.compile(input_types)
            return

        with Graph(compile_target.__name__, input_types=input_types) as graph:
            output = compile_target(*graph.inputs)
            graph.output(output)
            compiled_graph = graph

        if any(input_type.device.is_gpu() for input_type in input_types):
            device = Accelerator()
        else:
            device = CPU()
        session = InferenceSession([device])
        loaded_session = session.load(compiled_graph)
        self.session = loaded_session

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the compiled session with the given arguments.

        Args:
            *args: Positional arguments to pass to the session.
            **kwargs: Keyword arguments to pass to the session.

        Returns:
            The result of the session execution.
        """
        if self.is_module:
            return self.session(*args, **kwargs)
        return self.session.execute(*args, **kwargs)


def max_compile(
    compile_target: Callable | Module | None = None,
    input_types: Iterable[TensorType] | None = None,
) -> Callable[[Callable | Module], CompileWrapper] | CompileWrapper:
    """Decorator or function to compile a target with specified input types.

    Args:
        compile_target: The function or module to compile. If None, returns a decorator.
        input_types: The input types for the compilation.

    Returns:
        A CompileWrapper instance if compile_target is provided, otherwise a decorator.
    """
    if compile_target is None:

        def decorator(f: Callable | Module) -> CompileWrapper:
            return CompileWrapper(f, input_types)

        return decorator

    return CompileWrapper(compile_target, input_types)
