//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BinaryID.h"
#include "KGEN/CompilerRT/BinaryID.h"

MODULAR_CXX_EXPORT std::string M::KGEN::getCompilerRTBinaryID() {
  // M::getBinaryID() returns the binary ID of the shared library that contains
  // it. For the purposes of MEF cache invalidation, we need to know when
  // there's been a change in these shared libraries.
  return M::getBinaryID();
}
