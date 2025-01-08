//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/BuildInfo.h"
#include "Config/Version.h"
#include "Support/BinaryID.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace KGEN;

std::string M::KGEN::getVersionString() {
  return M::getModularVersionString() + M::getBinaryID() + "-" +
         M::getModularVersion().buildType;
}
