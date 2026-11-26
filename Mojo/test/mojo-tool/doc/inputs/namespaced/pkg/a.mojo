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

"""Defines a type that other modules in the package re-export."""


struct Thing(Copyable, Movable):
    """A documented type."""

    var value: Int
    """The value."""

    def __init__(out self, value: Int):
        """Initializes a thing.

        Args:
            value: The value.
        """
        self.value = value


def make_thing(value: Int) -> Thing:
    """Makes a thing.

    Args:
        value: The value.

    Returns:
        A new thing.
    """
    return Thing(value)
