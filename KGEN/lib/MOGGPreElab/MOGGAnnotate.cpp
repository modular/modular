//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Pass/Pass.h"

#include "Helpers.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_MOGGANNOTATE
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

static void annotateTypes(LIT::FuncOp func) {
  // Look through ref types to get underlaying decl ref type if needed.
  auto getAsDeclRefOrNull = [&](Type t) {
    auto asLitRef = dyn_cast<LIT::RefType>(t);
    if (asLitRef)
      return dyn_cast<LIT::DeclRefType>(asLitRef.getElementType());
    return dyn_cast<LIT::DeclRefType>(t);
  };

  // Anything taking a tensor needs the annotation.
  bool takesTensor = false;
  for (Type litType : func.getArgumentTypes()) {
    if (LIT::DeclRefType asDeclRef = getAsDeclRefOrNull(litType)) {
      takesTensor |= isMOGGTensor(asDeclRef);
      takesTensor |= isExtensibilityTensor(asDeclRef);
    }
  }

  if (!(isKernel(func) || isV1ShapeFunc(func) || takesTensor))
    return;

  OpBuilder builder{func.getContext()};

  SmallVector<Attribute> observedParams, typeNames, sourceName;
  observedParams.reserve(func.getNumArguments());

  Attribute emptyAttr = builder.getUnitAttr();

  // Extract the source name any of the lit argument.
  auto litTypeToSourceName = [&](Type litType) -> Attribute {
    LIT::DeclRefType asDeclRef = getAsDeclRefOrNull(litType);
    if (!asDeclRef)
      return emptyAttr;

    // We can't lower the symbol as it may become illegal at some point in IR so
    // we combine it into ROOT::LEAF;
    std::string combinedName =
        Twine(asDeclRef.getSymbol().getRootReference().strref())
            .concat("::")
            .concat(asDeclRef.getSymbol().getLeafReference().strref())
            .str();
    return builder.getStringAttr(combinedName);
  };

  // Extract the used parameters from the lit type.
  auto litTypeToParams = [&](Type litType) -> Attribute {
    LIT::DeclRefType asDeclRef = getAsDeclRefOrNull(litType);

    // We still need to have one entry per argument even if it is empty.
    if (!asDeclRef || asDeclRef.getParamValues().empty())
      return emptyAttr;

    SmallVector<Attribute> attrs;
    for (TypedAttr param : asDeclRef.getParamValues())
      attrs.push_back(param);
    return builder.getArrayAttr(attrs);
  };

  for (auto [i, litType] : llvm::enumerate(func.getArgumentTypes())) {
    observedParams.push_back(litTypeToParams(litType));
    typeNames.push_back(litTypeToSourceName(litType));

    sourceName.push_back(func.getSignature().getArgName(i));
  }

  // Attach the parameter mapping infomation to the kernel.
  if (!observedParams.empty()) {
    func->setDiscardableAttr(MOGG_ARG_PARAMS,
                             builder.getArrayAttr(observedParams));
  }

  // Add the result type.
  Type resultType = func.getResultTypes()[0];
  if (!isa<KGEN::NoneType>(resultType)) {
    func->setDiscardableAttr(MOGG_ARG_RESULT_PARAMS,
                             litTypeToParams(resultType));
  }

  if (!typeNames.empty()) {
    func->setDiscardableAttr(MOGG_ARG_TYPE_NAMES,
                             builder.getArrayAttr(typeNames));
  }

  if (!sourceName.empty()) {
    func->setDiscardableAttr(MOGG_ARG_SRC_NAMES,
                             builder.getArrayAttr(sourceName));
  }
}

namespace {
class MOGGAnnotatePass
    : public M::KGEN::MOGGPreElab::impl::MOGGAnnotateBase<MOGGAnnotatePass> {
public:
  void runOnOperation() override {
    ModuleOp op = getOperation();
    op.walk([](LIT::FuncOp func) {
      stripDecorators(func);
      annotateTypes(func);
    });
  }
};
} // namespace
