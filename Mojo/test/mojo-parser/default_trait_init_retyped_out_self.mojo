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

# RUN: %parse-mojo-isolated %s

# An `__init__` that retypes `out self` to an instantiation of its own struct
# must not make the struct's inherited default `__init__` resolve recursively.


trait MyDefaultable:
    def __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))


struct UsesDefault[n: Int](MyDefaultable):
    def __init__(out self, v: Int):
        pass


struct OverridesDefault[n: Int](MyDefaultable):
    def __init__(out self):
        pass

    def __init__(out self: OverridesDefault[1], v: Int):
        pass


def main():
    _ = UsesDefault[1](1)
    _ = OverridesDefault[1](1)
