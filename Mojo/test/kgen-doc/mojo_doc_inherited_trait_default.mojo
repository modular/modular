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
# RUN: kgen-doc %s | FileCheck %s

# An inheriting trait's copy of a default cannot be documented on its own, so
# its doc string has to come from the method the declaring trait defines. The
# copy carries `defaultFnRef` naming that method, and a copy of a copy keeps the
# original's, so this holds however many traits the default is inherited
# through.

"""Module docstring."""


trait Base:
    """A trait declaring a default."""

    def declared(self) -> Int:
        """Declared in Base.

        Returns:
            An Int.
        """
        return 1


# CHECK:      "summary": "Declared in Base."
# CHECK:      "kind": "trait"
# CHECK-NEXT: "name": "Base"


trait Middle(Base):
    """A trait one hop from the declaration."""

    pass


# CHECK:      "summary": "Declared in Base."
# CHECK:      "kind": "trait"
# CHECK-NEXT: "name": "Middle"


trait Leaf(Middle):
    """A trait two hops from the declaration."""

    pass


# CHECK:      "summary": "Declared in Base."
# CHECK:      "kind": "trait"
# CHECK-NEXT: "name": "Leaf"
