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
from max.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


from nn.attention.gpu.attn_dispatch import (
    AttnDispatchDesc,
    attn_dispatch_resolve as _attn_dispatch_resolve,
    mha_split_kv_decode_num_partitions_runtime,
    mla_decode_dispatch_scalars,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_num_partitions,
)


@export
def PyInit_kv_cache_ops() abi("C") -> PythonObject:
    """Creates a Python module with KV-cache helper bindings."""
    try:
        var b = PythonModuleBuilder("kv_cache_ops")
        b.def_function[mha_decode_num_partitions](
            "mha_decode_num_partitions",
            docstring="Returns the MHA decode partition count.",
        )
        b.def_function[mha_split_kv_decode_num_partitions](
            "mha_split_kv_decode_num_partitions",
            docstring=(
                "Returns the decode partition count for a split KV cache."
            ),
        )
        b.def_function[multiprocessor_count](
            "multiprocessor_count",
            docstring="Returns the device's multiprocessor count.",
        )
        b.def_function[mla_dispatch_args_scalar](
            "mla_dispatch_args_scalar",
            docstring=(
                "Returns the MLA dispatch metadata scalars and the kernel"
                " variant."
            ),
        )
        b.def_function[attn_dispatch_resolve](
            "attn_dispatch_resolve",
            docstring=(
                "Returns the decode dispatch of one KV cache: the key length,"
                " then the key, then the dispatch record scalars."
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create kv cache op bindings module: {e}")


def _make_int_list[n: Int](values: Array[Int, n]) -> PythonObject:
    ref cpython = Python().cpython()
    var result_py_list = cpython.PyList_New(len(values))
    for i in range(len(values)):
        _ = cpython.PyList_SetItem(
            result_py_list, i, cpython.PyLong_FromSsize_t(values[i])
        )
    return PythonObject(from_owned=result_py_list)


def _get_ctx(
    device_context_ptr: PythonObject,
) raises -> OpaquePointer[MutUntrackedOrigin]:
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        raise Error("Expected a non-null device context pointer.")
    return OpaquePointer[MutUntrackedOrigin](unsafe_from_address=addr)


def mha_decode_num_partitions(
    batch_size_obj: PythonObject,
    max_cache_valid_length_obj: PythonObject,
    n_kv_heads_obj: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var batch_size = Int(py=batch_size_obj)
    var max_cache_valid_length = Int(py=max_cache_valid_length_obj)
    var n_kv_heads = Int(py=n_kv_heads_obj)
    var ctx = _get_ctx(device_context_ptr)

    if batch_size < 1:
        raise Error("batch_size must be positive.")
    if max_cache_valid_length < 0:
        raise Error("max_cache_valid_length must be non-negative.")
    if n_kv_heads < 1:
        raise Error("n_kv_heads must be positive.")

    var device_ctx = DeviceContext(ctx)
    var num_partitions = mha_decoding_num_partitions(
        batch_size,
        max_cache_valid_length,
        n_kv_heads,
        device_ctx,
    )

    ref cpython = Python().cpython()
    return PythonObject(from_owned=cpython.PyLong_FromSsize_t(num_partitions))


def mha_split_kv_decode_num_partitions(
    batch_size_obj: PythonObject,
    num_keys_obj: PythonObject,
    kv_heads_per_device_obj: PythonObject,
    sliding_window_obj: PythonObject,
    sm_count_obj: PythonObject,
) raises -> PythonObject:
    var batch_size = Int(py=batch_size_obj)
    var num_keys = Int(py=num_keys_obj)
    var kv_heads_per_device = Int(py=kv_heads_per_device_obj)
    var sliding_window = Int(py=sliding_window_obj)
    var sm_count = Int(py=sm_count_obj)

    if batch_size < 1:
        raise Error("batch_size must be positive.")
    if num_keys < 0:
        raise Error("num_keys must be non-negative.")
    if kv_heads_per_device < 1:
        raise Error("kv_heads_per_device must be positive.")
    if sliding_window < 0:
        raise Error("sliding_window must be non-negative.")
    if sm_count < 1:
        raise Error("sm_count must be positive.")

    var num_partitions = mha_split_kv_decode_num_partitions_runtime(
        batch_size, num_keys, kv_heads_per_device, sliding_window, sm_count
    )
    ref cpython = Python().cpython()
    return PythonObject(from_owned=cpython.PyLong_FromSsize_t(num_partitions))


def multiprocessor_count(
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var device_ctx = DeviceContext(_get_ctx(device_context_ptr))
    var sm_count = device_ctx.device().multiprocessor_count()
    ref cpython = Python().cpython()
    return PythonObject(from_owned=cpython.PyLong_FromSsize_t(sm_count))


def mla_dispatch_args_scalar(
    batch_size_obj: PythonObject,
    max_cache_valid_length_obj: PythonObject,
    q_max_seq_len_obj: PythonObject,
    num_heads_obj: PythonObject,
    is_fp8_kv_obj: PythonObject,
    device_context_ptr: PythonObject,
    sparse_indices_stride_obj: PythonObject,
    kv_form_obj: PythonObject,
    page_size_obj: PythonObject,
    fixed_topk_obj: PythonObject,
    physical_indices_obj: PythonObject,
    depth_obj: PythonObject,
    row_invariant_verify_obj: PythonObject,
) raises -> PythonObject:
    var batch_size = Int(py=batch_size_obj)
    var max_cache_valid_length = Int(py=max_cache_valid_length_obj)
    var q_max_seq_len = Int(py=q_max_seq_len_obj)
    var num_heads = Int(py=num_heads_obj)
    var is_fp8_kv = Int(py=is_fp8_kv_obj) != 0
    var sparse_indices_stride = Int(py=sparse_indices_stride_obj)
    var kv_form = Int(py=kv_form_obj)
    var page_size = Int(py=page_size_obj)
    var fixed_topk = Int(py=fixed_topk_obj) != 0
    var physical_indices = Bool(py=physical_indices_obj)
    var depth = Int(py=depth_obj)
    var row_invariant_verify = Bool(py=row_invariant_verify_obj)
    var ctx = _get_ctx(device_context_ptr)

    if batch_size < 1:
        raise Error("batch_size must be positive.")
    if max_cache_valid_length < 0:
        raise Error("max_cache_valid_length must be non-negative.")
    if q_max_seq_len < 1:
        raise Error("q_max_seq_len must be positive.")
    if num_heads < 1:
        raise Error("num_heads must be positive.")
    if sparse_indices_stride < 0:
        raise Error("sparse_indices_stride must be non-negative.")
    if page_size < 0:
        raise Error("page_size must be non-negative.")

    var device_ctx = DeviceContext(ctx)

    var scalars = mla_decode_dispatch_scalars(
        device_ctx,
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        num_heads,
        is_fp8_kv,
        sparse_indices_stride,
        kv_form,
        page_size,
        fixed_topk,
        physical_indices,
        depth,
        row_invariant_verify,
    )
    var result: Array[Int, 4] = [
        scalars[0],
        scalars[1],
        scalars[2],
        scalars[3],
    ]
    return _make_int_list(result)


def attn_dispatch_resolve(
    device_context_ptr: PythonObject,
    kind_obj: PythonObject,
    fields_obj: PythonObject,
    batch_size_obj: PythonObject,
    q_max_seq_len_obj: PythonObject,
    max_cache_valid_length_obj: PythonObject,
) raises -> PythonObject:
    # `fields_obj` holds the descriptor's integer fields in declaration order.
    var desc = AttnDispatchDesc(
        kind=String(py=kind_obj),
        num_q_heads=Int(py=fields_obj[0]),
        num_kv_heads=Int(py=fields_obj[1]),
        head_dim=Int(py=fields_obj[2]),
        page_size=Int(py=fields_obj[3]),
        window_size=Int(py=fields_obj[4]),
        sparse_indices_stride=Int(py=fields_obj[5]),
        kv_dtype=Int(py=fields_obj[6]),
        kv_form=Int(py=fields_obj[7]),
        flags=Int(py=fields_obj[8]),
    )
    var resolved = _attn_dispatch_resolve(
        DeviceContext(_get_ctx(device_context_ptr)),
        desc,
        Int(py=batch_size_obj),
        Int(py=q_max_seq_len_obj),
        Int(py=max_cache_valid_length_obj),
    )
    ref cpython = Python().cpython()
    var count = 1 + len(resolved.key) + len(resolved.metadata)
    var result = cpython.PyList_New(count)
    _ = cpython.PyList_SetItem(
        result, 0, cpython.PyLong_FromSsize_t(len(resolved.key))
    )
    for i in range(len(resolved.key)):
        _ = cpython.PyList_SetItem(
            result, 1 + i, cpython.PyLong_FromSsize_t(resolved.key[i])
        )
    for i in range(len(resolved.metadata)):
        _ = cpython.PyList_SetItem(
            result,
            1 + len(resolved.key) + i,
            cpython.PyLong_FromSsize_t(resolved.metadata[i]),
        )
    return PythonObject(from_owned=result)
