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
"""Compatibility shim for ``asyncio.TaskGroup``.

``asyncio.TaskGroup`` was added in Python 3.11. On older interpreters we fall
back to the ``taskgroup`` PyPI backport. Importers should always use::

    from max.support.taskgroups import TaskGroup

so the version check and dependency live in a single place.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup
    from taskgroup import TaskGroup
else:
    BaseExceptionGroup = BaseExceptionGroup  # builtin in 3.11+
    from asyncio import TaskGroup

__all__ = ["BaseExceptionGroup", "TaskGroup"]
