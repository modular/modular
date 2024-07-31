//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MOGGPreElab/MOGGUtils.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "mlir/Pass/Pass.h"

#include "Helpers.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_MOGGLABELTENSORPARAMS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

static constexpr std::array<StringLiteral, 3> kMaxUnsafeTensorSlice = {
    "tensor_utils", "unsafe_tensor_slice", "UnsafeTensorSlice"};

namespace {
class MOGGLabelTensorParamsPass
    : public M::KGEN::MOGGPreElab::impl::MOGGLabelTensorParamsBase<
          MOGGLabelTensorParamsPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder{moduleOp.getContext()};
    MLIRContext *ctx = moduleOp.getContext();

    auto labelTensorParamsInKernel = [&](LIT::FuncOp funcOp) {
      // TODO: Replace with slariau's isDPSKernel.
      if (!funcOp->hasAttr(kMOGGExecuteFunctionLabel) &&
          !funcOp->hasAttr(kMOGGShapeFunctionLabel))
        return;

      // Look through ref types to get underlying decl ref type if needed.
      auto getAsStructType = [&](Type t) {
        auto asLitRef = dyn_cast<LIT::RefType>(t);
        if (asLitRef)
          return dyn_cast<LIT::StructType>(asLitRef.getElementType());
        return dyn_cast<LIT::StructType>(t);
      };

      // Extract the used parameters from the lit type.
      auto litTypeToParams = [&](LIT::StructType structType) {
        SmallVector<KGEN::ParamDeclRefAttr> attrs;
        for (TypedAttr param : structType.getParamValues()) {
          auto declRefAttr = dyn_cast<KGEN::ParamDeclRefAttr>(param);
          assert(declRefAttr);
          attrs.push_back(declRefAttr);
        }

        return attrs;
      };

      SmallVector<Attribute> tensorSpecs;
      Attribute emptyAttr = builder.getUnitAttr();
      for (auto [i, litType] : llvm::enumerate(funcOp.getArgumentTypes())) {
        auto asStructType = getAsStructType(litType);
        if (!asStructType ||
            !symbolMatches(asStructType.getSymbol(), kMaxUnsafeTensorSlice)) {
          tensorSpecs.push_back(emptyAttr);
          continue;
        }

        constexpr unsigned kDTypeIndex = 0;
        constexpr unsigned kRankIndex = 1;
        auto allParameters = litTypeToParams(asStructType);
        assert(allParameters.size() >= 2);
        auto dtype = allParameters[kDTypeIndex];
        auto rank = allParameters[kRankIndex];

        auto tensorSpecAttr = DictionaryAttr::get(
            ctx, {NamedAttribute{builder.getStringAttr("dtype"), dtype},
                  NamedAttribute{builder.getStringAttr("rank"), rank}});
        tensorSpecs.push_back(tensorSpecAttr);
      }
      funcOp->setDiscardableAttr(kKernelTensorParameterAttrName,
                                 builder.getArrayAttr(tensorSpecs));
    };

    moduleOp.walk(labelTensorParamsInKernel);
  }
};
} // namespace
