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

from __future__ import annotations

import atexit
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

from max import mlir

_DEFAULT_MLIR_CONTEXT = mlir.Context()
# MLIR Context.current is thread-local.
# - Keep the global context entered for the main thread.
# - Run worker tasks via helpers/executors in this module so MLIR APIs run
#   with a scoped context (avoids nanobind GIL/TLS teardown crashes).
# - atexit handler ensures the context is exited on shutdown.
_DEFAULT_MLIR_CONTEXT.__enter__()
atexit.register(_DEFAULT_MLIR_CONTEXT.__exit__, None, None, None)

P = ParamSpec("P")
R = TypeVar("R")


def default_mlir_context() -> mlir.Context:
    """Returns the global MLIR context."""
    return _DEFAULT_MLIR_CONTEXT


def call_with_default_mlir_context(
    fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> R:
    context = mlir.Context.current
    if context is None:
        with _DEFAULT_MLIR_CONTEXT:
            return fn(*args, **kwargs)
    return fn(*args, **kwargs)


class MLIRThreadPoolExecutor(ThreadPoolExecutor):
    def submit(
        self,
        fn: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R]:
        return super().submit(
            call_with_default_mlir_context, fn, *args, **kwargs
        )
