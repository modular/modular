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

from std.os import abort
from std.memory import OpaquePointer
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime.asyncrt import DeviceContextPtr

from nn.mha_decode_partition_heuristic import mha_decoding_num_partitions


@export
def PyInit_kv_cache_ops() -> PythonObject:
    """Creates a Python module with KV-cache helper bindings."""
    try:
        var b = PythonModuleBuilder("kv_cache_ops")
        b.def_function[mha_decode_num_partitions](
            "mha_decode_num_partitions",
            docstring="Returns the MHA decode partition count.",
        )
        return b.finalize()
    except e:
        abort(t"failed to create kv cache op bindings module: {e}")


@export
def mha_decode_num_partitions(
    batch_size_obj: PythonObject,
    max_cache_valid_length_obj: PythonObject,
    n_kv_heads_obj: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var batch_size = Int(py=batch_size_obj)
    var max_cache_valid_length = Int(py=max_cache_valid_length_obj)
    var n_kv_heads = Int(py=n_kv_heads_obj)
    var ctx = OpaquePointer[MutExternalOrigin](
        unsafe_from_address=Int(py=device_context_ptr)
    )

    if not ctx:
        raise Error("Expected a non-null device context pointer.")
    if batch_size < 1:
        raise Error("batch_size must be positive.")
    if max_cache_valid_length < 0:
        raise Error("max_cache_valid_length must be non-negative.")
    if n_kv_heads < 1:
        raise Error("n_kv_heads must be positive.")

    var device_ctx = DeviceContextPtr(ctx)
    var num_partitions = mha_decoding_num_partitions(
        batch_size,
        max_cache_valid_length,
        n_kv_heads,
        device_ctx.get_device_context(),
    )

    ref cpython = Python().cpython()
    return PythonObject(from_owned=cpython.PyLong_FromSsize_t(num_partitions))
