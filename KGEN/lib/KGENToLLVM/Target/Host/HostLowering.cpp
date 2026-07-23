//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MLIR-lowering policy for host (CPU) targets.
//
//===----------------------------------------------------------------------===//

#include "Target/Host/HostTraits.h"
#include "Target/TargetLowering.h"

namespace M::KGEN {
namespace {

class HostLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &HostTraits::get(); }

  // bast target that is always registered
  bool isBaseTarget() const override { return true; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<HostLowering> registerHostLowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
