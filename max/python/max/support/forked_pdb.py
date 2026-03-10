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

# Allow import of pdb. Ruff thinks this is debug code.
import pdb  # noqa: T100
import sys
from typing import Any


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child.

    This is particularly useful for adding breakpoints to the model worker
    MAX Serve sub-process. See: SERVSYS-941

    This is based on the implementation in:
    - https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess/23654936#23654936

    Usage:
    ```python
    # Start MAX Serve with `./bazelw run` or `br`
    from max.support import ForkedPdb
    ForkedPdb().set_trace()  # use in same way as `breakpoint()`
    ```
    """

    def interaction(self, *args: Any, **kwargs: Any) -> None:
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
