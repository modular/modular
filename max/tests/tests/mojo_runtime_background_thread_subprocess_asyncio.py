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
"""Subprocess entry: import ``max._interpreter_ops`` via ``asyncio.to_thread`` after ``InferenceSession``.

Executed via ``python …/mojo_runtime_background_thread_subprocess_asyncio.py`` from
``test_mojo_runtime_background_thread_import`` so the interpreter has no prior
MAX/Mojo imports from pytest.
"""

from __future__ import annotations

import asyncio

from max.driver import CPU
from max.engine import InferenceSession


async def _async_main() -> None:
    session = InferenceSession(devices=[CPU()])
    _ = session

    def import_ops() -> None:
        import max._interpreter_ops  # noqa: F401

    await asyncio.to_thread(import_ops)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
