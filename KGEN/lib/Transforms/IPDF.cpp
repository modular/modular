//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_IPDF
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct IPDF : impl::IPDFBase<IPDF> {
  using IPDFBase::IPDFBase;

  void runOnOperation() override;
};
} // namespace

void IPDF::runOnOperation() {}
