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


@value
@register_passable("trivial")
struct IO:
    var value: Int

    # TODO: either rename or get rid of this
    alias Unknown = IO(-1)

    alias Output = IO(0)
    alias Input = IO(1)

    # Represents the standard kind of fusion where we only make accesses
    # through the fusion lambda (e.g. any of the elementwise ops).
    alias FusedInput = IO(2)
    alias FusedOutput = IO(3)

    # Output fusion using a compute lambda.
    alias _FusedComputeOutput = IO(31)

    @always_inline("builtin")
    fn __init__(out self, value: Int):
        self.value = value

    fn __eq__(self, other: IO) -> Bool:
        return self.value == other.value


@value
@register_passable("trivial")
struct IOSpec[mut: Bool, input: IO]:
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


alias IOUnknown = IOSpec[True, IO.Unknown]()

alias Input = IOSpec[False, IO.Input]()
alias Output = IOSpec[True, IO.Output]()
alias MutableInput = IOSpec[True, IO.Input]()

alias FusedInput = IOSpec[False, IO.FusedInput]()
alias FusedOutput = IOSpec[True, IO.FusedOutput]()

alias _FusedComputeOutput = IOSpec[True, IO._FusedComputeOutput]()
