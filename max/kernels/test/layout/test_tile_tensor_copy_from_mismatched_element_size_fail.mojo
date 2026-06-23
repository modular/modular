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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var src_data = InlineArray[Int32, 4](fill=0)
    var dst_data = InlineArray[Int32, 8](fill=0)

    var src = TileTensor(src_data, row_major[2, 2]())
    var dst = TileTensor(dst_data, row_major[2, 4]()).vectorize[1, 2]()

    # CHECK: invalid call to 'copy_from': value passed to 'other' cannot be converted
    dst.copy_from(src)
