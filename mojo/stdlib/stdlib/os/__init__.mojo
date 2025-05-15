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
"""Provides access to operating-system dependent functionality.

The types and functions in this package primarily provide operating-system
independent access to operating-system dependent features, such as file systems
and environment variables.

For accessing files, see built-in [`open()`](/mojo/stdlib/builtin/file/open)
function and the [`file`](/mojo/stdlib/builtin/file/) module. For manipulating
file system paths, see the [`os.path`](/mojo/stdlib/os/path/) package for
OS-independent path manipulation functions and the `pathlib` package for
the [`Path`](/mojo/stdlib/pathlib/path/Path) struct, an abstraction for handling
paths.
"""

from .atomic import Atomic
from .env import getenv, setenv, unsetenv
from .fstat import lstat, stat, stat_result
from .os import (
    SEEK_CUR,
    SEEK_END,
    SEEK_SET,
    abort,
    getuid,
    listdir,
    makedirs,
    mkdir,
    remove,
    removedirs,
    rmdir,
    sep,
    unlink,
)
from .pathlike import PathLike
