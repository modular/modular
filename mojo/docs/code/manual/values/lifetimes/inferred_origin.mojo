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

from collections import List
from memory import Span


def to_byte_span[
    is_mutable: Bool, //,
    origin: Origin[is_mutable],
](ref [origin]list: List[Byte]) -> Span[Byte, origin]:
    return Span(list)


def main():
    list: List[Byte] = [77, 111, 106, 111]
    span = to_byte_span(list)
