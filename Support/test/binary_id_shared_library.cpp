//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BinaryID.h"
#include "Support/SymbolExport.h"
#include <string>

MODULAR_CXX_EXPORT std::string getSharedLibraryBinaryID() {
  return M::getBinaryID();
}
