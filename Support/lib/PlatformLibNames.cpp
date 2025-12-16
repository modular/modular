//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/PlatformLibNames.h"
#include "llvm/ADT/StringRef.h"
#include <string>

using namespace M;

std::string PlatformLibrary::getSharedLibraryName(llvm::StringRef name) {
  auto libName =
      std::string(SHARED_LIBRARY_PREFIX) + name.str() + SHARED_LIBRARY_SUFFIX;
  return libName;
}

std::string PlatformLibrary::getStaticLibraryName(llvm::StringRef name) {
  auto libName =
      std::string(STATIC_LIBRARY_PREFIX) + name.str() + STATIC_LIBRARY_SUFFIX;
  return libName;
}
