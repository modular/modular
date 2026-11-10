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
from max.gpu.host import DeviceAttribute, DeviceContext
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_mla_decode_dispatch_scalars,
    mha_decoding_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    compute_mla_dispatch_key_runtime,
)
from nn.attention.gpu.nvidia.sm100.dispatch import (
    mha_split_kv_decode_num_partitions as _split_kv_decode_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.fa4_splitk_combine import P_MAX


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

    # The policy's ladder tops out at the SM count, a compile-time parameter,
    # so it is instantiated for every count up to the combine's `P_MAX`, which
    # also caps the count it returns. A MIG slice gets its own count's ladder.
    # TODO(DISTINF-639): a device with more than `P_MAX` SMs takes the `P_MAX`
    # instantiation, whose wave bound counts `P_MAX` SMs and so may split less
    # than the device could; give the policy a runtime SM count if one ships.
    var policy_sm_count = min(sm_count, P_MAX)
    comptime for ladder_sm_count in range(1, P_MAX + 1):
        if policy_sm_count == ladder_sm_count:
            var num_partitions = _split_kv_decode_num_partitions[
                ladder_sm_count
            ](batch_size, num_keys, kv_heads_per_device, sliding_window)
            ref cpython = Python().cpython()
            return PythonObject(
                from_owned=cpython.PyLong_FromSsize_t(num_partitions)
            )
    raise Error("unreachable: every SM count up to P_MAX has a policy")


def multiprocessor_count(
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var device_ctx = DeviceContext(_get_ctx(device_context_ptr))
    var sm_count = device_ctx.device().get_attribute(
        DeviceAttribute.MULTIPROCESSOR_COUNT
    )
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

    # The variant separates capture keys whose launches share a split count;
    # only the SM100 sparse decode has more than one.
    var scalars: Tuple[Int, Int, Int, Int]
    if device_ctx.api() == "hip":
        var amd = amd_mla_decode_dispatch_scalars(
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            is_fp8_kv,
            device_ctx,
            sparse_indices_stride=sparse_indices_stride,
        )
        scalars = (amd[0], amd[1], amd[2], 0)
    elif device_ctx.api() == "cuda":
        scalars = compute_mla_dispatch_key_runtime(
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            is_fp8_kv,
            sparse_indices_stride,
            device_ctx.device().get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            ),
            kv_form=kv_form,
            page_size=page_size,
            fixed_topk=fixed_topk,
            physical_indices=physical_indices,
            depth=depth,
        )
    else:
        raise Error(
            "MLA decode dispatch metadata supports only HIP (AMD) and CUDA"
            " (NVIDIA) devices."
        )
    var result: Array[Int, 4] = [
        scalars[0],
        scalars[1],
        scalars[2],
        scalars[3],
    ]
    return _make_int_list(result)
