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
#include "KGEN/MOGGPreElab/MOGGUtils.h"
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

static constexpr llvm::StringLiteral kExecuteFuncName = "execute";
static constexpr llvm::StringLiteral kShapeFuncName = "shape";

static void annotateTypes(LIT::FuncOp func) {
  // Look through ref types to get underlaying decl ref type if needed.
  auto getAsDeclRefOrNull = [&](Type t) {
    auto asLitRef = dyn_cast<LIT::RefType>(t);
    if (asLitRef)
      return dyn_cast<LIT::StructType>(asLitRef.getElementType());
    return dyn_cast<LIT::StructType>(t);
  };

  // Anything taking a tensor needs the annotation.
  bool takesTensor = false;
  for (Type litType : func.getArgumentTypes()) {
    if (LIT::StructType asDeclRef = getAsDeclRefOrNull(litType)) {
      takesTensor |= isMOGGTensor(asDeclRef);
      takesTensor |= isExtensibilityTensor(asDeclRef);
      takesTensor |= isDPSTensor(asDeclRef);
    }
  }

  if (!isKernel(func) && !isV1ShapeFunc(func) && !isDPSKernel(func) &&
      !takesTensor)
    return;

  OpBuilder builder{func.getContext()};

  SmallVector<Attribute> observedParams, typeNames, sourceName;
  observedParams.reserve(func.getNumArguments());

  Attribute emptyAttr = builder.getUnitAttr();

  // Extract the source name any of the lit argument.
  auto litTypeToSourceName = [&](Type litType) -> Attribute {
    LIT::StructType asDeclRef = getAsDeclRefOrNull(litType);
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
    LIT::StructType asDeclRef = getAsDeclRefOrNull(litType);

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
    OpBuilder builder{op.getContext()};

    // Do a first walk through the IR to strip the decorators and add
    // attributes.
    op->walk([](Operation *operation) {
      if (auto func = dyn_cast<LIT::FuncOp>(operation)) {
        stripDecorators(func);
        annotateTypes(func);
      } else if (auto structDeclOp = dyn_cast<LIT::StructDeclOp>(operation)) {
        stripDecorators(structDeclOp);
      }
    });

    // Do another walk to complete the annotation.
    // We need two walks because some op X might look at the annotations of op
    // Y, which might be defined after X. (Eg check if a decorator function
    // has a mogg_intrinsic_attr set).
    const auto walker = [&](Operation *operation) {
      if (auto structDeclOp = dyn_cast<LIT::StructDeclOp>(operation)) {
        auto loc = structDeclOp->getLoc();
        std::optional<StringAttr> registrationName;
        auto decorators = structDeclOp.getDecorators();
        if (decorators.empty())
          return WalkResult::advance();

        // Iterate over the decorators and to find max.compiler.register.
        for (auto decorator : decorators) {
          auto apply = dyn_cast<ParamOperatorAttr>(decorator);
          if (!apply || apply.getNumOperands() != 2)
            continue;

          auto sym = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
          if (!sym)
            continue;

          auto decoratorFunc = op.lookupSymbol<LIT::FuncOp>(sym.getSymbol());
          if (!decoratorFunc ||
              !decoratorFunc->hasAttr(MOGG_INTRINSIC_REGISTER))
            continue;

          auto [_, nameAttr] =
              cast<LIT::LITStructAttr>(apply.getOperand(1)).getValues().front();
          auto name = dyn_cast<StringAttr>(nameAttr);
          if (!name)
            continue;

          if (registrationName) {
            emitError(loc, "Only one op can be registered per kernel struct");
            return WalkResult::interrupt();
          }
          registrationName = name;
        }

        if (!registrationName)
          return WalkResult::advance();

        // Search for the `execute` and the optional `shape` functions. Until
        // traits can enforce @staticmethod, this has to be done in this pass.
        LIT::FuncOp executeFunc, shapeFunc;
        for (auto &op : structDeclOp.getFields().front()) {
          auto func = dyn_cast<LIT::FuncOp>(op);
          if (!func)
            continue;

          if (func.getSourceName() == kExecuteFuncName)
            executeFunc = func;
          else if (func.getSourceName() == kShapeFuncName)
            shapeFunc = func;
        }

        if (!executeFunc) {
          emitError(loc, "Kernels must have an execution entry point named " +
                             kExecuteFuncName);
          return WalkResult::interrupt();
        }

        if (!executeFunc.getIsStatic()) {
          emitError(loc, "Kernel entry point must be a static function");
          return WalkResult::interrupt();
        }

        if (shapeFunc && !shapeFunc.getIsStatic()) {
          emitError(loc, "Kernel shape function must be a static function");
          return WalkResult::interrupt();
        }

        executeFunc->setAttr(builder.getStringAttr(kMOGGExecuteFunctionLabel),
                             *registrationName);
        if (shapeFunc) {
          shapeFunc->setAttr(builder.getStringAttr(kMOGGShapeFunctionLabel),
                             *registrationName);
        }

        for (auto param : executeFunc.getInputParams()) {
          if (param.getName() == kMOGGSynchronousParameterName) {
            executeFunc->setAttr(builder.getStringAttr(kMOGGSynchronousLabel),
                                 param);
          } else if (param.getName() == kMOGGTargetParameterName) {
            executeFunc->setAttr(builder.getStringAttr(kMOGGTargetLabel),
                                 param);
          }
        }
      }

      return WalkResult::advance();
    };

    if (op.walk(walker).wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
