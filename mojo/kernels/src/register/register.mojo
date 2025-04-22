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


fn register_internal(name: StaticString):
    """
    This decorator registers a given mojo function as being an implementation
    of a mo op or a `mo.custom` op.

    For instance:

    @register_internal("mo.add")
    fn my_op[...](...):

    registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
    """
    return


fn __mogg_intrinsic_attr(intrin: StaticString):
    """
    Attaches the given intrinsic annotation onto the function.
    """
    return
