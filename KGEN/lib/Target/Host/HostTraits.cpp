//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Target/Host/HostTraits.h"

namespace M::KGEN {

const HostTraits &HostTraits::get() {
  static const HostTraits instance;
  return instance;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<HostTraits> registerHostTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
