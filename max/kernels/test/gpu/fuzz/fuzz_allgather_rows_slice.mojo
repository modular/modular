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
"""Fuzzes row-sliced allgather against the original host input bytes."""

from std.collections import Array
from std.math import gcd
from std.random import seed
from std.sys import size_of
from std.sys.defines import get_defined_int

from comm import Signal
from comm.allgather import allgather_rows_slice
from comm.sync import enable_p2p, init_signal_buffer
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from _fuzz import boundary_int, collect_args, flag, flag_int

comptime ngpus = get_defined_int["slice_allgather_gpus", 2]()
comptime dtype = (
    DType.float16 if get_defined_int[
        "slice_allgather_fp16", 0
    ]() else DType.bfloat16
)


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var rows_per_rank: Int
    var col_units: Int
    var repeats: Int
    var order: Int
    var data_seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "rows_per_rank=",
            self.rows_per_rank,
            " col_units=",
            self.col_units,
            " repeats=",
            self.repeats,
            " order=",
            self.order,
            " data_seed=",
            self.data_seed,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for i in range(n):
        var rows = boundary_int(0, 64, 8)
        var cols = boundary_int(0, 4096, 1024)
        if i % 6 == 0:
            rows = 1
            cols = 1022 + (i // 6) % 3
        elif i % 6 == 1:
            rows = 0
        elif i % 6 == 2:
            rows = 8
            cols = boundary_int(0, 64, 8)
        specs.append(
            CaseSpec(
                rows,
                cols,
                boundary_int(0, 5, 2),
                i % 3,
                boundary_int(0, 65535, 256),
            )
        )
    return specs^


def fill_bits(
    host: HostBuffer[dtype], data_seed: Int, source: Int, iteration: Int
):
    var ptr = host.unsafe_ptr().unsafe_bitcast[UInt16]()
    var state = UInt64(data_seed + 1 + source * 65537 + iteration * 31337)
    for i in range(len(host)):
        state = state * 6364136223846793005 + 1442695040888963407
        var bits = UInt16(state >> 48)
        if i % 16 == 0:
            bits = 0x8000
        elif i % 16 == 1:
            comptime if dtype == .bfloat16:
                bits = 0x7FC1
            else:
                bits = 0x7E01
        elif i % 16 == 2:
            comptime if dtype == .bfloat16:
                bits = 0x7F80
            else:
                bits = 0x7C00
        elif i % 16 == 3:
            bits = 1
        ptr[unsafe_offset=i] = bits


def run_one_case(contexts: List[DeviceContext], spec: CaseSpec) raises:
    if not (
        0 <= spec.rows_per_rank <= 64
        and 0 <= spec.col_units <= 4096
        and 0 <= spec.repeats <= 5
        and 0 <= spec.order <= 2
        and 0 <= spec.data_seed <= 65535
    ):
        raise Error("spec is outside the supported fuzz domain")
    # Encoding alignment in the spec keeps every shrink candidate legal.
    var cols = (8 // gcd(8, max(1, spec.rows_per_rank))) * (spec.col_units + 1)
    var slice_elems = spec.rows_per_rank * cols
    var total = ngpus * slice_elems
    var inputs = List[DeviceBuffer[dtype]](capacity=ngpus)
    var hosts = List[HostBuffer[dtype]](capacity=ngpus)
    var outputs = List[List[DeviceBuffer[dtype]]](capacity=ngpus)
    var signals = List[DeviceBuffer[.uint8]](capacity=ngpus)
    for d in range(ngpus):
        inputs.append(contexts[d].create_buffer_sync[dtype](total))
        hosts.append(contexts[d].enqueue_create_host_buffer[dtype](total))
        signals.append(
            contexts[d].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(signals[d], contexts[d])
        var row = List[DeviceBuffer[dtype]](capacity=ngpus)
        for _ in range(ngpus):
            row.append(contexts[d].create_buffer_sync[dtype](slice_elems))
        outputs.append(row^)

    comptime InTile = type_of(
        TileTensor(inputs[0], row_major(total)).as_imm().as_unsafe_any_origin()
    )
    comptime OutTile = type_of(
        TileTensor(outputs[0][0], row_major(slice_elems)).as_unsafe_any_origin()
    )

    def signal_at(i: Int) {mut signals} -> MutPointer[Signal, MutAnyOrigin]:
        return Signal.unsafe_ptr_from(signals[i])

    var signal_ptrs = Array[_, ngpus](fill_with=signal_at)

    for iteration in range(spec.repeats + 1):
        var empty = iteration % 3 == 1
        var active_total = 0 if empty else total
        var active_slice = 0 if empty else slice_elems

        def input_at(i: Int) {mut inputs, imm active_total} -> InTile:
            return (
                TileTensor(inputs[i], row_major(active_total))
                .as_imm()
                .as_unsafe_any_origin()
            )

        var in_tensors = Array[_, ngpus](fill_with=input_at)

        def output_at(i: Int) {mut outputs, imm active_slice} -> OutTile:
            var d, source = divmod(i, ngpus)
            return TileTensor(
                outputs[d][source], row_major(active_slice)
            ).as_unsafe_any_origin()

        var out_tensors = Array[_, ngpus * ngpus](fill_with=output_at)
        for source in range(ngpus):
            fill_bits(hosts[source], spec.data_seed, source, iteration)
            contexts[source].enqueue_copy(inputs[source], hosts[source])
        for d in range(ngpus):
            for source in range(ngpus):
                var sentinel = contexts[d].enqueue_create_host_buffer[dtype](
                    slice_elems
                )
                var ptr = sentinel.unsafe_ptr().unsafe_bitcast[UInt16]()
                var expected = (
                    hosts[source].unsafe_ptr().unsafe_bitcast[UInt16]()
                )
                for k in range(slice_elems):
                    ptr[unsafe_offset=k] = ~expected[
                        unsafe_offset=d * slice_elems + k
                    ]
                contexts[d].enqueue_copy(outputs[d][source], sentinel)
                contexts[d].synchronize()
        for i in range(ngpus):
            var d = i
            if spec.order == 1:
                d = ngpus - 1 - i
            elif spec.order == 2:
                d = (i + iteration + 1) % ngpus
            allgather_rows_slice(
                in_tensors, out_tensors, signal_ptrs, cols, contexts[d], d
            )
        for d in range(ngpus):
            contexts[d].synchronize()
        for d in range(ngpus):
            for source in range(ngpus):
                var result = contexts[d].enqueue_create_host_buffer[dtype](
                    slice_elems
                )
                contexts[d].enqueue_copy(result, outputs[d][source])
                contexts[d].synchronize()
                var actual = result.unsafe_ptr().unsafe_bitcast[UInt16]()
                var expected = (
                    hosts[source].unsafe_ptr().unsafe_bitcast[UInt16]()
                )
                for k in range(slice_elems):
                    var wanted = expected[unsafe_offset=d * slice_elems + k]
                    if empty:
                        wanted = ~wanted
                    if actual[unsafe_offset=k] != wanted:
                        print(
                            t"FUZZ_NUMERIC_FAIL destination= {d} source="
                            t" {source} element= {k} iteration= {iteration}"
                        )
                        raise Error("sliced allgather changed the source bytes")
    _ = inputs
    _ = hosts
    _ = outputs
    _ = signals


def main() raises:
    comptime assert ngpus in (2, 4, 8), "use 2, 4 or 8 GPUs"
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    seed(flag_int(args, "--seed", 12345))
    var budget = flag_int(args, "--budget", 16)
    if mode == "list-specs":
        var specs = gen_specs(budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, specs[i])
        return
    if DeviceContext.number_of_devices() < ngpus:
        raise Error("not enough visible GPUs for sliced allgather")
    if not enable_p2p():
        raise Error("sliced allgather fuzzing requires P2P access")
    var contexts = List[DeviceContext](capacity=ngpus)
    for d in range(ngpus):
        contexts.append(DeviceContext(device_id=d))
    if mode == "single":
        run_one_case(
            contexts,
            CaseSpec(
                flag_int(args, "--rows_per_rank", 1),
                flag_int(args, "--col_units", 1023),
                flag_int(args, "--repeats", 2),
                flag_int(args, "--order", 2),
                flag_int(args, "--data_seed", 12345),
            ),
        )
        print("FUZZ_RESULT verdict=PASS")
        return
    if mode != "fuzz":
        raise Error("unknown fuzz mode")
    var specs = gen_specs(budget)
    for i in range(len(specs)):
        run_one_case(contexts, specs[i])
    print("FUZZ_RESULT verdict=PASS")
