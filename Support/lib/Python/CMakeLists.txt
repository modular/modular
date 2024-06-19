##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)
include(RTTIExceptions)
include(AddPyBind11)


pybind11_add_module(MPython STATIC
  Helpers.cpp
)

set_rtti_exceptions(MPython)
