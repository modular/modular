##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-repl-entry-point
  main.cpp

  LINK_LIBS PUBLIC
  KGENCompilerRT
  CUDASupportGlobals
  AsyncRTRuntimeGlobals
  MSupportGlobals
  MSupport
  MAsyncRTInit
)

# The entry point is a win32 executable, and uses WinMain instead of main. This
# removes the console window that would otherwise be created when starting the
# REPL in LLDB.
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  set_target_properties(mojo-repl-entry-point PROPERTIES WIN32_EXECUTABLE TRUE)

  set_property(TARGET mojo-repl-entry-point
    PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
  target_compile_definitions(mojo-repl-entry-point
    PRIVATE MODULAR_BUILDING_LIBRARY
  )
endif()

# This target is supposed to be located next to libMojoLLDB.
set_target_properties(mojo-repl-entry-point PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
)

# We need to prevent inline function expansion to guarentee our entry
# point function is correctly invoked.
if(MSVC)
  target_compile_options(mojo-repl-entry-point PRIVATE /Od)
else()
  set_target_properties(mojo-repl-entry-point PROPERTIES
      COMPILE_FLAGS "-g0 -O0")
endif()
