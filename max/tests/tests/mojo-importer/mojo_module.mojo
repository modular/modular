# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from python import PythonObject
from python.bindings import PythonModuleBuilder

from os import abort


@export
fn PyInit_mojo_module() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module")
        m.def_function[plus_one]("plus_one")

        return m.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


fn plus_one(arg: PythonObject) raises -> PythonObject:
    return arg + 1
