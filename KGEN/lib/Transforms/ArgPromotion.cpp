//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/KGENPasses.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_ARGPROMOTION
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class ArgPromotionPass : public impl::ArgPromotionBase<ArgPromotionPass> {
public:
  using ArgPromotionBase::ArgPromotionBase;
  void runOnOperation() override;
};
} // namespace

void ArgPromotionPass::runOnOperation() {}
