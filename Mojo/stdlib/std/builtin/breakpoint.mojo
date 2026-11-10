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
"""This module includes the builtin breakpoint function."""

from std.sys import breakpointhook


@inline(.nodebug)
def breakpoint():
    """Triggers a debugger breakpoint at the call site.

    Execution stops here and control passes to the attached debugger, so the
    program state can be inspected at this point. If no debugger is
    attached, the process terminates with a trap signal.

    For example:

    ```mojo
    def process(value: Int):
        if value < 0:
            breakpoint()  # stop and inspect why the value is negative
    ```
    """
    breakpointhook()
