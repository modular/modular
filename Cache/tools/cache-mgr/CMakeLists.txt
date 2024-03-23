##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(additional_depends)
if (MODULAR_WITH_S3_CAS)
  set(additional_depends blobcache_s3)
endif()

add_modular_tool(cache-mgr
  cache-mgr.cpp

  DEPENDS
  ${additional_depends}

  LINK_LIBS PRIVATE
  MCache
  MSupport
  MSupportGlobals
  MSupportInit
  MLIRSupport
  LLCLRuntimeGlobals
  )
