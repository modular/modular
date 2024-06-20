##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)

modular_find_python3_development()

include(RTTIExceptions)
include(AddPyBind11)


add_library(MPython STATIC
  Helpers.cpp
)

target_include_directories(MPython PRIVATE "${Python3_INCLUDE_DIRS}" "${pybind11_INCLUDE_DIRS}")

set_rtti_exceptions(MPython)
