# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.memory import OwnedPointer
from weights_registry import WeightsRegistry


@fieldwise_init
struct Result(TrivialRegisterPassable):
    var _value: Int32

    comptime SUCCESS = Self(0)
    comptime FAILURE = Self(1)


# C API fallible constructor for a weights registry.
# Sets the output `weights` to a Mojo WeightsRegistry containing the name: data
# mapping.
# `names` is an array of null-terminated C-style strings.
# `data` is an array of pointers to buffers backing the weight.
# The number of names and data pointers should be equal, and names[i] is the
# name of the weight backed by data[i].
@export
def create_weights_registry(
    weights: Pointer[OpaquePointer[MutUntrackedOrigin], MutUntrackedOrigin],
    names: Pointer[Pointer[UInt8, ImmUntrackedOrigin], MutUntrackedOrigin],
    data: Pointer[OpaquePointer[ImmUntrackedOrigin], MutUntrackedOrigin],
    size: Int,
) abi("C") -> Result:
    var weights_dict = Dict[String, OpaquePointer[ImmUntrackedOrigin]](
        capacity=size
    )
    for i in range(size):
        var name = String(unsafe_from_utf8_ptr=names[unsafe_offset=i])
        weights_dict[name] = data[unsafe_offset=i]

    # Emplace a new weights registry into the output pointer.
    var ptr = OwnedPointer(WeightsRegistry(weights_dict^))
    weights[] = (
        ptr^.unsafe_take_allocation().unsafe_leak().unsafe_bitcast[NoneType]()
    )
    return Result.SUCCESS


@export
def destroy_weights_registry(
    weights: MutOpaquePointer[MutUntrackedOrigin],
) abi("C") -> None:
    _ = OwnedPointer[WeightsRegistry](unsafe_from_opaque_pointer=weights)


# Looks up a weight in the weights registry.
# Returns Result.FAILURE if not found.
@export
def weights_registry_lookup(
    data: Pointer[OpaquePointer[ImmUntrackedOrigin], MutUntrackedOrigin],
    weights: OpaquePointer[MutUntrackedOrigin],
    name: Pointer[UInt8, ImmUntrackedOrigin],
) abi("C") -> Result:
    var weights_ptr = weights.unsafe_bitcast[WeightsRegistry]()
    try:
        data[] = weights_ptr[][String(unsafe_from_utf8_ptr=name)]
    except e:
        return Result.FAILURE

    return Result.SUCCESS


# Looks up a weight in the weights registry using a (pointer to a) Mojo String.
# Returns Result.FAILURE if not found.
@export
def weights_registry_lookup_mojo_string(
    data: Pointer[OpaquePointer[ImmUntrackedOrigin], MutUntrackedOrigin],
    weights: OpaquePointer[MutUntrackedOrigin],
    name: Pointer[String, ImmUntrackedOrigin],
) abi("C") -> Result:
    var weights_ptr = weights.unsafe_bitcast[WeightsRegistry]()
    try:
        data[] = weights_ptr[][name[]]
    except e:
        return Result.FAILURE

    return Result.SUCCESS
