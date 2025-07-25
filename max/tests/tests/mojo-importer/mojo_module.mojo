# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from python._cpython import GILAcquired, GILReleased
from os import abort
import math
from algorithm.functional import parallelize
from sys.info import num_physical_cores


@export
fn PyInit_mojo_module() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[plus_one]("plus_one")
        m.def_function[parallel_wrapper](
            "parallel_wrapper", docstring="Parallelizing function"
        )
        return m.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


fn plus_one(arg: PythonObject) raises -> PythonObject:
    return arg + 1


fn parallel_wrapper(array: PythonObject) raises -> PythonObject:
    alias do_parallelize = True
    var array_len = len(array)
    var array_len_div = math.ceildiv(array_len, num_physical_cores())

    @parameter
    fn calc_max(i: Int) -> None:
        ref cpython = Python().cpython()
        # Each worker needs to hold the GIL to access python objects.
        # It is more efficient to only use Mojo native data structures in worker threads.
        with GILAcquired(Python(cpython)):
            try:
                var start_idx = i * array_len_div
                var end_idx = min((i + 1) * array_len_div, array_len)

                var max_val = array[start_idx]
                for j in range(start_idx + 1, end_idx):
                    if array[j] > max_val:
                        max_val = array[j]

                array[start_idx] = max_val
            except e:
                pass

    ref cpython = Python().cpython()

    @parameter
    if do_parallelize:
        # Save the current thread state to avoid holding the GIL for the parallel loop.
        with GILReleased(Python(cpython)):
            parallelize[calc_max](num_physical_cores())
    else:
        for i in range(0, num_physical_cores()):
            calc_max(i)

    var final_max = array[0]
    for i in range(1, num_physical_cores()):
        if array[i * array_len_div] > final_max:
            final_max = array[i * array_len_div]
    array[0] = final_max

    return array
