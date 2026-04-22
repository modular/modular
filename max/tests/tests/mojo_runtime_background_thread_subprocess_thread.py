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
"""Subprocess entry: import ``max._interpreter_ops`` on a ``threading.Thread`` after ``InferenceSession``.

Executed via ``python …/mojo_runtime_background_thread_subprocess_thread.py`` from
``test_mojo_runtime_background_thread_import`` so the interpreter has no prior
MAX/Mojo imports from pytest.
"""

from __future__ import annotations

import threading

from max.driver import CPU
from max.engine import InferenceSession


def main() -> None:
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            import max._interpreter_ops  # noqa: F401
        except BaseException as exc:
            errors.append(exc)

    session = InferenceSession(devices=[CPU()])
    _ = session

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=120.0)
    assert not thread.is_alive(), (
        "import max._interpreter_ops on background thread timed out"
    )
    assert not errors, f"Mojo interpreter ops import failed: {errors[0]!r}"


if __name__ == "__main__":
    main()
