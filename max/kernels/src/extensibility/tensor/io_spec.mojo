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


struct IO(TrivialRegisterPassable):
    var value: SIMDSize

    # TODO: either rename or get rid of this
    comptime Unknown = IO(-1)

    comptime Output = IO(0)
    comptime Input = IO(1)

    # Represents the standard kind of fusion where we only make accesses
    # through the fusion lambda (e.g. any of the elementwise ops).
    comptime FusedInput = IO(2)
    comptime FusedOutput = IO(3)

    # Output fusion using a compute lambda.
    comptime _FusedComputeOutput = IO(31)

    @always_inline("builtin")
    def __init__(out self, value: Int):
        self.value = value

    def __eq__(self, other: IO) -> Bool:
        return self.value == other.value

    def __ne__(self, other: IO) -> Bool:
        return self.value != other.value

    @always_inline("nodebug")
    def is_fused(self) -> Bool:
        """True when this IO represents any fused variant (input, output, or
        compute-output)."""
        return (
            self == IO.FusedInput
            or self == IO.FusedOutput
            or self == IO._FusedComputeOutput
        )


@fieldwise_init
struct IOSpec[mut: Bool, input: IO](TrivialRegisterPassable):
    """
    Parameter used to encode whether a particular tensor argument to a DPS kernel
    is an output, input, or mutable input.

    ```mojo
    Input == IOSpec[False, IO.Input]()
    Output == IOSpec[True, IO.Output]()
    MutableInput == IOSpec[True, IO.Input]()
    FusedInput == IOSpec[False, IO.FusedInput]()
    FusedOutput == IOSpec[True, IO.FusedOutput]()
    ```
    """

    ...


comptime IOUnknown = IOSpec[True, IO.Unknown]()

comptime Input = IOSpec[False, IO.Input]()
comptime Output = IOSpec[True, IO.Output]()
comptime MutableInput = IOSpec[True, IO.Input]()

comptime FusedInput = IOSpec[False, IO.FusedInput]()
comptime FusedOutput = IOSpec[True, IO.FusedOutput]()

comptime _FusedComputeOutput = IOSpec[True, IO._FusedComputeOutput]()
