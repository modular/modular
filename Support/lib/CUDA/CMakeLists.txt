##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(CUDAToolkit QUIET)

set(NVTX_LIB_FOUND OFF)
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND MODULAR_LLC_PROFILING_NSIGHT)
  if(${CMAKE_VERSION} VERSION_LESS 3.25)
    if(TARGET CUDA::nvToolsExt)
      set(NVTX_LIB_FOUND ON)
      set(NVTX_LIB CUDA::nvToolsExt)
    endif()
  elseif(TARGET CUDA::nvtx3)
    set(NVTX_LIB_FOUND ON)
    set(NVTX_LIB CUDA::nvtx3)
  endif()
endif()

if (NVTX_LIB_FOUND)
  add_compile_definitions(USE_NVTX_LIB)
  add_compile_definitions(NVTX_LIBRARY_PATH="${CUDA_nvToolsExt_LIBRARY}")
endif()

add_modular_library(MSupportCUDA
  NVTX.cpp

  LINK_LIBS PUBLIC
  MSupport
)
