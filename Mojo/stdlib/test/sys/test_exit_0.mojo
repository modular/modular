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

import std.sys


def exit_or_value(should_exit: Bool) -> Int:
    # `exit` returns `Never`, so this function needs no trailing `return`
    # (https://github.com/modular/modular/issues/7297).
    if not should_exit:
        return 1
    std.sys.exit(0)


def main():
    _ = exit_or_value(False)
    _ = exit_or_value(True)
