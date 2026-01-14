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

from pathlib import Path
from tempfile import gettempdir


fn write_to_disk(path: String, data: String) raises:
    """Write string to file."""
    var p = Path(path)
    with open(p, "w") as f:
        f.write(data)


@explicit_destroy("Must call save_and_close() or discard()")
struct FileBuffer:
    var path: String
    var data: String

    fn __init__(out self, path: String):
        self.path = path
        self.data = ""

    fn write(mut self, content: String):
        self.data += content

    fn save_and_close(deinit self) raises:
        write_to_disk(self.path, self.data)

    fn discard(deinit self):
        # Abandon buffered data without writing
        pass


fn main() raises:
    var path = gettempdir().or_else("/tmp/") + "test-write.txt"
    var buffer = FileBuffer(path)
    buffer.write("Hello from FileBuffer\n")
    buffer^.save_and_close()  # Comment out for compiler error

    print("File written successfully to", path)
