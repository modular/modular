##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(additional_depends)

add_modular_tool(cache-mgr
  cache-mgr.cpp

  DEPENDS
  ${additional_depends}

  LINK_LIBS PRIVATE
  MCache
  MSupport
  MSupportGlobals
  Init
  MLIRSupport
  AsyncRTRuntimeGlobals
  )
