##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(RTTIExceptions)
include(AddPyBind11)

add_library(MPython STATIC
  Helpers.cpp
)

target_include_directories(MPython PRIVATE "${pybind11_INCLUDE_DIRS}")

set_rtti_exceptions(MPython)
