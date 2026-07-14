//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Target/Host/HostTraits.h"

#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

const HostTraits &HostTraits::get() {
  static const HostTraits instance;
  return instance;
}

llvm::StringRef HostTraits::defaultCPU(const llvm::Triple &triple) const {
  // 32-bit ARM (arm/armeb) has no host-CPU default; pin it to the arch's
  // baseline CPU.
  if (triple.isARM())
    return llvm::ARM::getDefaultCPU(triple.getArchName());
  return {};
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<HostTraits> registerHostTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
