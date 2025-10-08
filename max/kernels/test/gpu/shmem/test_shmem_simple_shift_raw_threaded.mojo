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
# RUN: %mojo-build %s -o %t
# RUN: %t

from gpu.host import DeviceContext, DeviceBuffer
from algorithm import parallelize
from shmem import *
from shmem._nvshmem import *
from testing import assert_equal


fn simple_shift_kernel(destination: UnsafePointer[Int32]):
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes
    print("GPU mype:", mype, "peer:", peer)

    shmem_p(destination, mype, peer)


# This test shows you how to run your own initialization and finalization logic
# instead of using the higher-level `shmem_launch` function. See
# `test_shmem_simple_shift.mojo` for the more ergonomic version of this.
def simple_shift(mype_node: Int):
    # SHMEM initialization that runs once per attached GPU, using MPI
    # bootstrapping to handle internode communication.
    var ctx = DeviceContext(mype_node)
    ctx.set_as_current()

    # Get MPI_COMM_WORLD symbol and set
    var mpi_comm = get_mpi_comm_world()
    var attr = NVSHMEMXInitAttr(UnsafePointer(to=mpi_comm))

    # Get the number of attached GPUs on this node
    var npes_node = DeviceContext.number_of_devices()

    # Set the args with correct values values for this node
    attr.args.uid_args.myrank = c_int(mype_node)
    attr.args.uid_args.nranks = c_int(npes_node)
    var nvshmem_status = nvshmemx_hostlib_init_attr(
        NVSHMEMX_INIT_WITH_MPI_COMM, UnsafePointer(to=attr)
    )
    if nvshmem_status:
        raise Error(
            "failed to initialize NVSHMEM with error code:", nvshmem_status
        )

    # Compile the function and set device state in const memory
    var func = ctx.compile_function[simple_shift_kernel]()
    shmem_module_init(func)

    # Stores the pe of the device that communicated with this PE
    var target_device = shmem_malloc[DType.int32](1)
    var target_host = ctx.enqueue_create_host_buffer[DType.int32](1)

    # Launch the function and synchronize across all devices
    ctx.enqueue_function(func, target_device, grid_dim=1, block_dim=1)
    shmem_barrier_all_on_stream(ctx.stream())

    # Copy the pe that communicated with this PE back to host
    target_host.enqueue_copy_from(target_device)
    ctx.synchronize()

    # Get the device mype and npes across all nodes to test communication
    # working correctly between devices.
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + npes - 1) % npes
    assert_equal(target_host[0], peer)

    # Host-side deinitialization
    shmem_free(target_device)
    shmem_module_finalize(func)
    shmem_finalize()


def main():
    # All setup logic that runs single threaded on device 0 here
    var npes_node = DeviceContext.number_of_devices()

    # Enable running across multiple nodes with `mpirun`
    var _argv = argv()
    var argc = len(_argv)
    MPI_Init(argc, _argv)

    # Enable any exceptions inside `simple_shift` to abort, `parallelize` can't
    # run on raising functions.
    fn wrap_simple_shift(mype_node: Int) capturing:
        try:
            simple_shift(mype_node)
        except e:
            abort(String("SHMEM failed on mype_node: ", mype_node, ": ", e))

    # Everything inside simple_shift runs in parallel per attached GPU on this node
    # Same number of tasks as worker threads
    parallelize[wrap_simple_shift](npes_node, npes_node)

    # Cleanup MPI resources
    MPI_Finalize()
