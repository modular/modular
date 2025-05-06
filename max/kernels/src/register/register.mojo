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
"""Provides APIs for registering MAX Graph operations."""


fn register_internal(name: StaticString):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op or a `mo.custom` op. This decorator is used for built-in
    [MAX Graph operations](/max/api/python/graph/ops).

    For registering [custom operations](/max/custom-ops/), use the
    [@compiler.register](/mojo/manual/decorators/compiler-register) decorator,
    instead.

    For instance:

    ```mojo
    @register_internal("mo.add")
    fn my_op[...](...):
      ...
    ```

    Registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
    """
    return


fn __mogg_intrinsic_attr(intrin: StaticString):
    """
    Attaches the given intrinsic annotation onto the function.
    """
    return
