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

from std.collections.string import StaticString
from std.os import abort

from std.memory import OwnedPointer
from std.memory.unsafe_pointer import unsafe_cast

from max.runtime.tracing import Color, Trace, TraceLevel

# Set C API events to TraceLevel.OP to emit nvtx events.
# Otherwise, with TraceLevel.ALWAYS, AsyncRT events are emitted.
comptime OpTrace = Trace[TraceLevel.OP]
# Safety: can't have inferred parameters with exported C ABI functions
# so we set AnyOrigin here, which will still extend the lifetime of
# any structs that pass their pointers to the below functions. Don't
# use UntrackedOrigin here or it will turn off extending the lifetime
# of Mojo structs that are responsible for deallocating the pointer.
comptime TracePointer = OptionalPointer[NoneType, AnyOrigin[mut=True]]
comptime NestedTracePointer = OptionalPointer[TracePointer, AnyOrigin[mut=True]]


@fieldwise_init
struct Result(ImplicitlyCopyable, RegisterPassable):
    var _value: Int32

    comptime SUCCESS = Self(0)
    comptime FAILURE = Self(1)


@export
def create_trace(
    trace: NestedTracePointer,
    message: StaticString,
    color: StaticString,
) abi("C") -> Result:
    # Emplace a new trace object into the output pointer.
    # TODO(KERN-1052): amortize multiple calls with string registration.
    var ptr = OwnedPointer(OpTrace(message, color=Color(color)))

    # Set the output.
    var storage = ptr^.unsafe_take_allocation().unsafe_leak()
    trace.value()[] = storage.unsafe_bitcast[NoneType]().as_unsafe_any_origin()

    return Result.SUCCESS


@export
def destroy_trace(trace: TracePointer) abi("C"):
    if case .Some(trace) = trace:
        _ = OwnedPointer[OpTrace](
            unsafe_from_opaque_pointer=trace.unsafe_origin_cast[
                UntrackedOrigin[mut=True]
            ]()
        )


@export
def start_trace(trace: TracePointer) abi("C"):
    var trace_ptr = unsafe_cast[Type=OpTrace](trace)

    # Start the profiling span.
    try:
        trace_ptr.value()[].start()
    except e:
        abort(String(e))


@export
def end_trace(trace: TracePointer) abi("C"):
    var trace_ptr = unsafe_cast[Type=OpTrace](trace)

    # End the profiling span.
    try:
        trace_ptr.value()[].end()
    except e:
        abort(String(e))


@export
def mark_trace(trace: TracePointer) abi("C"):
    var trace_ptr = unsafe_cast[Type=OpTrace](trace)

    # Mark the profile span.
    try:
        trace_ptr.value()[].mark()
    except e:
        abort(String(e))
