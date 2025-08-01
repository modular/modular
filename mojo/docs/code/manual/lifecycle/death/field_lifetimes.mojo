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


@fieldwise_init
struct Balloons:
    var color: String
    var count: Int


fn consume(var arg: String):
    pass


fn use(arg: Balloons):
    print(arg.count, arg.color, "balloons.")


fn consume_and_use():
    var balloons = Balloons("blue", 8)
    consume(balloons.color^)
    # String.__moveinit__() runs here, which invalidates balloons.color
    # Now balloons is only partially initialized

    # use(balloons)  # This fails because balloons.color is uninitialized

    balloons.color = String("orange")  # All together now
    use(balloons)  # This is ok
    # balloons.__del__() runs here (and only if the object is whole)


def main():
    var balloons = Balloons("red", 5)
    print(balloons.color)
    # balloons.color.__del__() runs here, because this instance is
    # no longer used; it's replaced below

    balloons.color = "blue"  # Overwrite balloons.color
    print(balloons.color)
    # balloons.__del__() runs here

    # second example
    consume_and_use()
