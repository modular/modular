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


struct Item(Copyable, Movable):
    var x: Int


# Nothing else in the program names `List[Optional[Item]]`, so its layout is
# first lowered from the type of this field.
struct Backend(Copyable, Movable):
    var make: def() thin -> List[Optional[Item]]


def backend_is_none() -> Bool:
    var backend = Optional[Backend](None)
    return not backend
