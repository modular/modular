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
"""IO constants and functions."""
from sys.ffi import _Global
from builtin.io import _fdopen

alias stdin = FileDescriptor(0)
alias STDIN = _Global["STDIN", _fdopen["r"], _init_stdin]


fn _init_stdin() -> _fdopen["r"]:
    return _fdopen["r"](stdin)


alias stdout = FileDescriptor(1)
alias stderr = FileDescriptor(2)
