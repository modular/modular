//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/BuildInfo.h"
#include "Config/Version.h"
#include "Support/BinaryID.h"

using namespace M;
using namespace KGEN;

std::string M::KGEN::getVersionString() {
  return M::getMojoVersionString() + M::getBinaryID() + "-" +
         M::getMojoVersion().buildType;
}
