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


@fieldwise_init
struct Grid(Copyable, Writable):
    var rows: Int
    var cols: Int
    var data: List[List[Int]]

    def write_to(self, mut writer: Some[Writer]):
        # Iterate through rows 0 through rows-1
        for row in range(self.rows):
            # Iterate through columns 0 through cols-1
            for col in range(self.cols):
                if self.data[row][col] == 1:
                    # If cell is populated, write an asterisk
                    writer.write_string("*")
                else:
                    # If cell is not populated, write a space
                    writer.write_string(" ")
            if row != self.rows - 1:
                # Write a newline between rows, but not at the end
                writer.write_string("\n")
