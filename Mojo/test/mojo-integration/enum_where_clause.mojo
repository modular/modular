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

# RUN: %mojo %s | FileCheck %s

"""A `where` clause may compare an `EnumLike` parameter to an enum case.

The case value is a compile-time expression that cannot be reduced to a simple
constant, so the comparison is decided by the identity of the case rather than
by calling `__eq__`.
"""


__enum State(Equatable):
    case created
    case configured


struct Handle[state: State]:
    """A handle whose state is fixed by its `state` parameter."""

    def __init__(out self) where Self.state == State.created:
        pass


def main():
    var handle = Handle[.created]()
    _ = handle^
    # CHECK: created
    print("created")
