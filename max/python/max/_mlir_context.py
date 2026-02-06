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
import contextlib
from collections.abc import Generator

from max import mlir

_DEFAULT_MLIR_CONTEXT = mlir.Context()
# MLIR Context.current is thread-local.
# - Keep the global context entered for the main thread.
# - Use active_mlir_context() in worker threads so MLIR APIs run with a
#   scoped context (avoids nanobind GIL/TLS teardown crashes).
# - atexit handler ensures the context is exited on shutdown.
_DEFAULT_MLIR_CONTEXT.__enter__()
atexit.register(_DEFAULT_MLIR_CONTEXT.__exit__, None, None, None)


def default_mlir_context() -> mlir.Context:
    """Returns the global MLIR context."""
    return _DEFAULT_MLIR_CONTEXT


@contextlib.contextmanager
def active_mlir_context() -> Generator[mlir.Context]:
    context = mlir.Context.current
    if context is None:
        with _DEFAULT_MLIR_CONTEXT:
            yield _DEFAULT_MLIR_CONTEXT
    else:
        yield context
