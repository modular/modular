//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/Threading.h"
#include "mlir/IR/Matchers.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ApplyInlinerPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_APPLYINLINER
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ApplyInlinerPass : impl::ApplyInlinerBase<ApplyInlinerPass> {
  using ApplyInlinerBase::ApplyInlinerBase;
  void runOnOperation() override;
};

struct FunctionTrait {
  static std::optional<FunctionTrait> identify(GeneratorOp func);

  /// This trait represents a function that trivially forwards a single
  /// register-passable argument.
  ///
  /// ```mlir
  /// kgen.generator @anything(%arg0: !SomeType) -> !SomeType {
  ///   kgen.return %arg0 : !SomeType
  /// }
  /// ```
  ///
  /// This means we can peephole `apply(@anything, x) -> x` regardless of
  /// parameter bindings.
  struct RegForward {};

  /// This trait represents a function that trivially forwards a single
  /// register-passable argument through a result slot, either `init_self` or
  /// `byref_result`.
  ///
  /// ```mlir
  /// kgen.generator @anything(%result: !kgen.pointer<!SomeType> init_self,
  ///                          %value: !SomeType) -> !kgen.none {
  ///   pop.store %value, %result
  ///   kgen.return %none
  /// }
  /// ```
  ///
  /// This means we can peephole `apply_result_slot(@anything, x) -> x`
  /// regardless of parameter bindings.
  struct RegSlotForward {};

  /// This trait represents a function that simply returns a constant value with
  /// no side effects.
  ///
  /// ```mlir
  /// kgen.generator @constant() -> !SomeType {
  ///   %0 = kgen.param.constant: !SomeType = <...>
  ///   kgen.return %0 : !SomeType
  /// }
  /// ```
  ///
  /// This means we can peephole `apply(@constant) -> value`, while substituting
  /// any parameter values.
  struct RegConstant {
    ArrayRef<ParamDeclAttr> params;
    Attribute value;
  };

  SmartVariant<RegForward, RegSlotForward, RegConstant> impl;
};
} // namespace

std::optional<FunctionTrait> FunctionTrait::identify(GeneratorOp func) {
  // In all patterns, the function is terminated by a return.
  auto ret = dyn_cast<ReturnOp>(func.getBody()->getTerminator());
  if (!ret)
    return {};
  SignatureType sig = func.getSignature();
  Operation *first = &func.getBody()->front();

  // Check one argument, one result, return is only operation in the body. It
  // follows that the return operand must be the function argument, since there
  // is nothing else it can be.
  if (sig.getNumArguments() == 1 && ret.getNumOperands() == 1 && ret == first)
    return FunctionTrait{RegForward{}};

  // Check zero arguments, one result, the first operation is a constant, return
  // is the only other operation. It follows that the return operand must be the
  // constant.
  auto cst = dyn_cast<ParamConstantOp>(first);
  if (cst && sig.getNumArguments() == 0 && ret.getNumOperands() == 1 &&
      cst->getNextNode() == ret)
    return FunctionTrait{RegConstant{func.getInputParams(), cst.getValue()}};

  // Check two arguments, one result, the return value is a none constant, and
  // one of the first two ops is a store, and the third op is a return.
  NoneAttr noneAttr;
  POP::StoreOp store;
  if (sig.getNumArguments() == 2 && ret.getNumOperands() == 1 &&
      mlir::matchPattern(ret.getOperand(0), mlir::m_Constant(&noneAttr)) &&
      ((store = dyn_cast<POP::StoreOp>(first)) ||
       (store = dyn_cast<POP::StoreOp>(first->getNextNode()))) &&
      (store->getNextNode() == ret ||
       ret.getOperand(0).getDefiningOp()->getNextNode() == ret)) {
    int valueIdx = -1;
    if (sig.getArgConvention(0) == ArgConvention::InitSelf)
      valueIdx = 1;
    else if (sig.getArgConvention(1) == ArgConvention::ByRefResult)
      valueIdx = 0;
    // Check that one of the arguments is a result slot, the result slot
    // argument is the dest of the store, and the other argument is the value.
    if (valueIdx != -1 && store.getPtr() == func.getArgument(!valueIdx) &&
        store.getArg() == func.getArgument(valueIdx))
      return FunctionTrait{RegSlotForward{}};
  }

  return {};
}

void ApplyInlinerPass::runOnOperation() {
  DenseMap<StringAttr, FunctionTrait> funcTraits;
  for (auto func : getOperation().getOps<GeneratorOp>())
    if (std::optional<FunctionTrait> trait = FunctionTrait::identify(func))
      funcTraits.try_emplace(func.getSymNameAttr(), std::move(*trait));

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&funcTraits](ParamOperatorAttr apply) -> TypedAttr {
    if (apply.getOpcode() != POC::Apply &&
        apply.getOpcode() != POC::ApplyResultSlot)
      return apply;
    auto cst = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
    if (!cst)
      return apply;
    auto it =
        funcTraits.find(cast<FlatSymbolRefAttr>(cst.getSymbol()).getAttr());
    if (it == funcTraits.end())
      return apply;
    FunctionTrait trait = it->second;

    // Handle RegSlotForward.
    if (apply.getOpcode() == POC::ApplyResultSlot) {
      assert(isa<FunctionTrait::RegSlotForward>(trait.impl));
      return apply.getOperand(1);
    }

    // Handle RegForward.
    if (isa<FunctionTrait::RegForward>(trait.impl))
      return apply.getOperand(1);

    // Handle RegConstant.
    auto regCst = cast<FunctionTrait::RegConstant>(trait.impl);
    ParameterEvaluator evaluator(regCst.params, cst.getParamValues());
    return cast<TypedAttr>(evaluator.getReboundAttribute(regCst.value));
  });

  // The replacers have an internal cache, so make sure to share them correctly.
  auto substTrivialFuncs = [](mlir::AttrTypeReplacer &replacer, Operation *op) {
    replacer.recursivelyReplaceElementsIn(
        op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);
  };
  std::vector<Operation *> ops;
  for (Operation &op : getOperation().getOps())
    ops.push_back(&op);
  parallelForEach(&getContext(), ops, substTrivialFuncs, replacer);
}
