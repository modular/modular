//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BinaryID.h"
#include "binary_id_shared_library.h"

#include <iostream>

int main() {
  std::cout << "binary id: " << M::getBinaryID() << "\n";
  std::cout << "shared library id: " << getSharedLibraryBinaryID() << "\n";
}
