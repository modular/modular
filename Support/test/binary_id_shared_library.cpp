//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BinaryID.h"
#include "Support/SymbolExport.h"

MODULAR_CXX_EXPORT std::string getSharedLibraryBinaryID() {
  return M::getBinaryID();
}
