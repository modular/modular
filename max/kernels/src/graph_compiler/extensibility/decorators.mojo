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
"""Decorators for registering MAX Graph kernels.

`@register` is the public decorator for DPS-style custom kernels.
`@register_internal` is used by built-in MAX Graph operations.
"""


def register_internal(name: StaticString):
    """
    Registers the given mojo function as an implementation of an `mo` op or a
    `mo.custom` op. Used by built-in
    [MAX Graph operations](/max/api/python/graph.ops).

    For registering [custom operations](/max/develop/custom-ops/), use the
    [@compiler.register](/mojo/manual/decorators/compiler-register) decorator,
    instead.

    For instance:

    ```mojo
    @register_internal("mo.add")
    def my_op[...](...):
      ...
    ```

    Registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
    """
    return


# Register a DPS Kernel.
def register(name: StaticString):
    pass


# Indicates that a DPS Kernel is a view operation.
def view_kernel():
    return
