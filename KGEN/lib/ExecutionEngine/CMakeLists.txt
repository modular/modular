##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVMCompilerRTLibPath ${LLVM_BUILD_DIR}/lib/clang/${LLVM_VERSION_MAJOR})
if (APPLE)
  set(ORCSuffix "_osx")
  set(LLVMCompilerRTLibPath ${LLVMCompilerRTLibPath}/lib/darwin)
elseif (WIN32)
  set(ORCSuffix "-x86_64")
  set(LLVMCompilerRTLibPath ${LLVMCompilerRTLibPath}/lib/windows)
elseif (UNIX)
  set(ORCSuffix "")

  # Get the architecture.
  set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
  if (arch STREQUAL "x86")
    set(arch "i686")
  endif ()
  # Get the vendor.
  set(vendor "unknown")
  # Get os.
  string(TOLOWER ${CMAKE_HOST_SYSTEM_NAME} os)
  set(triple "${arch}-${vendor}-${os}-gnu")
  message(STATUS "Host triple: ${triple}")
  set(LLVMCompilerRTLibPath ${LLVMCompilerRTLibPath}/lib/${triple})
endif ()

list(APPEND CMAKE_PREFIX_PATH ${LLVMCompilerRTLibPath})
find_library(ORC_LIB orc_rt${ORCSuffix} REQUIRED)

cache_files(orc-rt-cas
  BASE_DIR ${MODULAR_DERIVED_DIR}/.kgen_cache/orc
  HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/ORCCASID.h
  INPUTS ${ORC_LIB}
  INPUT_NAMES "MODULAR_ORC_RT_CAS_ID")

add_modular_mlir_library(KGENExecutionEngine
  ExecutionEngine.cpp

  LINK_COMPONENTS
  OrcJIT
  OrcShared
  OrcTargetProcess
  JITLink
  MC

  DEPENDS
  orc-rt-cas

  LINK_LIBS PUBLIC
  KGENDialect
  KGENToLLVM
  KGENEmitObject

  MCache

  MLIRLLVMToLLVMIRTranslation
  MLIRIndexToLLVM
  MLIRTargetLLVMIRExport
  )
