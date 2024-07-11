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

find_library(ORC_LIB orc_rt${ORCSuffix} REQUIRED
  HINTS "${MODULAR_SOURCE_DIR}/${LLVMCompilerRTLibPath}")

add_modular_library(KGENExecutionEngine
  ExecutionEngine.cpp

  DEPENDS
  KGENCompilerRTShared
  KGENVersion

  LINK_COMPONENTS
  OrcDebugging
  OrcJIT
  OrcShared
  OrcTargetProcess
  JITLink
  MC

  LINK_LIBS PUBLIC
  KGENJIT
  KGENSupport
  AsyncRTRuntime
  MCache
  MSupport
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND TARGET LLVMPerfJITEvents)
  target_link_libraries(KGENExecutionEngine
    PUBLIC
    LLVMPerfJITEvents
    LLVMOrcTargetProcess
  )
endif()

set_property(TARGET KGENExecutionEngine PROPERTY KGEN_ORC_RT_LIB "${ORC_LIB}")
