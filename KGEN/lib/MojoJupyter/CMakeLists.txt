##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Generate MatplotlibInitialization.h, which contains the initialization code
# to run to initialize matplotlib in the kernel.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS enable_matplotlib.py)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS matplotlib_backend.py)
file(READ enable_matplotlib.py ENABLE_MATPLOTLIB_STR)
file(READ matplotlib_backend.py MATPLOTLIB_BACKEND_STR)

set(matplotlib_inc "${MODULAR_BINARY_DIR}/KGEN/include/KGEN/MojoJupyter/MatplotlibInitialization.h")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MatplotlibInitialization.h.in ${matplotlib_inc}
@ONLY)
set_source_files_properties("${matplotlib_inc}"
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)

add_modular_library(MojoJupyter SHARED
  Kernel.cpp
  ${matplotlib_inc}

  LINK_LIBS PRIVATE
  KGENSupport
  MSupportGlobals
  MSupport
  MojoLLDB

  MLIRIR
  liblldb
)

# Ensure symbols from the plugin library are visible.
set_target_properties(MojoJupyter PROPERTIES CXX_VISIBILITY_PRESET default)
target_compile_definitions(MojoJupyter PRIVATE MODULAR_BUILDING_JUPYTER_LIBRARY)
