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


# start-noop-destructor
@fieldwise_init
struct Balloons:
    var color: String
    var count: Int

    fn __del__(deinit self):
        # Mojo destroys all the fields when they're last used
        pass
        # end-noop-destructor


# start-basic-destructor
@fieldwise_init
struct Balloon(Writable):
    var color: String

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(String("a ", self.color, " balloon"))

    fn __del__(deinit self):
        print("Destroyed", String(self))


def main():
    var a = Balloon("red")
    var b = Balloon("blue")
    print(a)
    # a.__del__() runs here for "red" Balloon

    a = Balloon("green")
    # a.__del__() runs immediately because "green" Balloon is never used

    print(b)
    # b.__del__() runs here
    # end-basic-destructor
    _ = a^
