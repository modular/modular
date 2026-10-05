//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This pass lowers a variety of high level Mojo types in the 'lit' dialect
// to lower level KGEN abstractions.  Notably, this eliminates symbol based
// struct references (in favor of `!kgen.struct`), `!lit.ref` => `!kgen.pointer`
// etc.  This runs immediately after the LowerLIT pass.
//
//===----------------------------------------------------------------------===//

#include "LowerLITTypes.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Config/Version.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Struct Layout Dependency Analysis
//===----------------------------------------------------------------------===//
//
// A graph that captures struct references.

namespace {
struct StructRefNode {
  StringAttr name;
  /// The parameter positions the struct holds by value. A position counts when
  /// building the struct's layout requires building the layout of the argument
  /// in it: `@Box<ty> { v: !kgen.param<ty> }` has one, while
  /// `@Ptr<ty> { p: !kgen.pointer<ty> }` has none, because the pointee is
  /// erased.
  llvm::BitVector byValueParams;
};

/// One node per declared struct.
struct StructRefGraph {
  explicit StructRefGraph(StructDecls &decls) {
    // Reserve up front: `byName` points into `nodes`.
    nodes.reserve(decls.structDecls.size());
    for (auto &[name, decl] : decls.structDecls) {
      StructRefNode &node = nodes.emplace_back();
      node.name = name;
      node.byValueParams.resize(decl.decls ? decl.decls.size() : 0);
      byName[name] = &node;
    }
  }
  StructRefGraph(const StructRefGraph &) = delete;

  StructRefNode *lookup(StringAttr name) const { return byName.lookup(name); }

  std::vector<StructRefNode> nodes;
  DenseMap<StringAttr, StructRefNode *> byName;
};

class StructDependencyAnalysis;

/// Walks a struct's field types, recording on its node which of its own
/// parameters it holds by value and which structs it references.
class StructDependencyWalker {
public:
  StructDependencyWalker(ParamDeclArrayAttr params,
                         StructDependencyAnalysis &analysis,
                         StructRefGraph &graph, StructRefNode &self)
      : analysis(analysis), graph(graph), self(self) {
    if (params) {
      for (auto [index, param] : llvm::enumerate(params.getValue()))
        paramIndex[param.getName()] = index;
    }
  }

  void walk(Type type, bool indirect) {
    if (!type || !visit(type.getAsOpaquePointer(), indirect))
      return;
    // A pointer field's pointee is replaced with `none` and never substituted,
    // so nothing reachable through it can constrain this layout.
    if (auto ptr = dyn_cast<PointerType>(type)) {
      walk(ptr.getElementType(), /*indirect=*/true);
      walk(ptr.getAddressSpace(), indirect);
      return;
    }
    // A function-typed field also lowers to a pointer.
    if (isa<FuncTypeGeneratorType>(type)) {
      walkSubElements(type, /*indirect=*/true);
      return;
    }
    if (auto ref = dyn_cast<LIT::StructType>(type)) {
      walkStructRef(ref, indirect);
      return;
    }
    walkSubElements(type, indirect);
  }

  void walk(Attribute attr, bool indirect) {
    if (!attr || !visit(attr.getAsOpaquePointer(), indirect))
      return;
    if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr)) {
      if (!indirect)
        markByValue(paramRef.getName());
      // A parameter's declared type is not part of the layout of the struct
      // declaring it, so there is nothing further to walk here.
      return;
    }
    walkSubElements(attr, indirect);
  }

private:
  /// Types and attributes are uniqued into a DAG, so a shared subterm can be
  /// reached many ways.
  bool visit(const void *ptr, bool indirect) {
    return visited[indirect].insert(ptr).second;
  }

  void markByValue(StringAttr name) {
    auto it = paramIndex.find(name);
    // Not a parameter of this struct: it belongs to some nested signature.
    if (it == paramIndex.end())
      return;
    if (it->second < self.byValueParams.size())
      self.byValueParams.set(it->second);
  }

  void walkStructRef(LIT::StructType ref, bool indirect);

  void walkSubElements(Type type, bool indirect) {
    type.walkImmediateSubElements([&](Attribute attr) { walk(attr, indirect); },
                                  [&](Type type) { walk(type, indirect); });
  }
  void walkSubElements(Attribute attr, bool indirect) {
    attr.walkImmediateSubElements([&](Attribute attr) { walk(attr, indirect); },
                                  [&](Type type) { walk(type, indirect); });
  }

  StructDependencyAnalysis &analysis;
  StructRefGraph &graph;
  StructRefNode &self;
  DenseMap<StringAttr, unsigned> paramIndex;
  /// Indexed by the indirection state the subterm was reached under.
  DenseSet<const void *> visited[2];
};

/// Fills the struct graph over every declaration, rejecting by-value recursion
/// on the way.
///
/// Each struct is walked once, after a depth-first visit of the structs it
/// depends on. The one kind of dependency that cannot be met is a struct still
/// being visited and which is a struct that holds, by value, something that
/// holds it by value. This represents a layout with no finite size.
class StructDependencyAnalysis {
public:
  StructDependencyAnalysis(StructDecls &decls, StructRefGraph &graph)
      : decls(decls), graph(graph) {}

  /// Walks every declaration. Fails once every by-value cycle is reported.
  LogicalResult run() {
    for (StructRefNode &node : graph.nodes) {
      if (!state.contains(node.name))
        visit(node);
    }
    return failure(anyIllegal);
  }

  /// Called by the walk of the struct on top of the stack for a struct it holds
  /// by value, before the arguments passed to it are walked.
  void markDependsOnByValue(StructRefNode &node) {
    auto it = state.find(node.name);
    if (it == state.end()) {
      visit(node);
    } else if (it->second == State::Visiting) {
      anyIllegal = true;
      ArrayRef<StringAttr> cycle(stack);
      cycle = cycle.drop_front(llvm::find(cycle, node.name) - cycle.begin());
      // One report per struct is enough to act on.
      if (llvm::any_of(cycle,
                       [&](StringAttr s) { return reported.contains(s); }))
        return;
      reported.insert(cycle.begin(), cycle.end());
      auto diag = mlir::emitError(decls.get(node.name).loc)
                  << "'" << node.name.getValue()
                  << "' must not contain itself by value; store the recursive "
                     "field behind a pointer";
      for (StringAttr hop : cycle.drop_front()) {
        diag.attachNote(decls.get(hop).loc)
            << "'" << hop.getValue() << "' holds its parameter by value";
      }
    }
  }

private:
  enum class State { Visiting, Done };

  void visit(StructRefNode &node) {
    state[node.name] = State::Visiting;
    stack.push_back(node.name);
    StructDecl &decl = decls.get(node.name);
    StructDependencyWalker walker(decl.decls, *this, graph, node);
    for (Type field : llvm::make_second_range(decl.fields))
      walker.walk(field, /*indirect=*/false);
    stack.pop_back();
    state[node.name] = State::Done;
  }

  StructDecls &decls;
  StructRefGraph &graph;
  DenseMap<StringAttr, State> state;
  SmallVector<StringAttr> stack;
  DenseSet<StringAttr> reported;
  bool anyIllegal = false;
};

void StructDependencyWalker::walkStructRef(LIT::StructType ref, bool indirect) {
  StructRefNode *target = graph.lookup(ref.getName());
  assert(target && "Reference to unknown struct");

  // Under an indirection nothing constrains the layout, so which positions the
  // struct holds by value is beside the point.
  if (indirect) {
    for (TypedAttr arg : ref.getParamValues())
      walk(arg, /*indirect=*/true);
    return;
  }

  // Held by value, so an argument to a position the struct holds by value is
  // held by value here too. Its positions are final once it has been visited.
  analysis.markDependsOnByValue(*target);
  const llvm::BitVector &argByValue = target->byValueParams;
  for (auto [index, arg] : llvm::enumerate(ref.getParamValues())) {
    bool byValueArg = index < argByValue.size() && argByValue.test(index);
    walk(arg, !byValueArg);
  }
}

} // namespace

/// Rejects struct layouts that contain themselves by value, which have no
/// finite size.
static LogicalResult analyzeStructRefs(StructDecls &decls) {
  StructRefGraph graph(decls);
  return StructDependencyAnalysis(decls, graph).run();
}

namespace {
/// A thin layer over `mlir::CyclicAttrTypeReplacer`, which reports failure as
/// a null result. It is surfaced here as `FailureOr` so no caller can drop it.
class Lowering {
public:
  FailureOr<Attribute> replace(Attribute element) {
    if (Attribute result = impl.replace(element))
      return result;
    return failure();
  }
  FailureOr<Type> replace(Type element) {
    if (Type result = impl.replace(element))
      return result;
    return failure();
  }

  /// Replace the attributes of `op`, and optionally its location and its
  /// result and block-argument types. Unlike the walk upstream provides, this
  /// fails on the first element that does not lower rather than leaving it in
  /// place, so the pass reports an error instead of emitting LIT.
  LogicalResult replaceElementsIn(Operation *op, bool replaceAttrs = true,
                                  bool replaceLocs = false,
                                  bool replaceTypes = false);

  /// Add a rule that skips recursing down its own result. The callback does
  /// any further replacing itself, which is what lets one lowering hand a
  /// sub-element to another.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>,
            typename ResultT = std::invoke_result_t<FnT, T>>
  std::enable_if_t<std::is_convertible_v<ResultT, FailureOr<BaseT>>>
  addRule(FnT &&callback) {
    impl.addReplacement(mlir::CyclicAttrTypeReplacer::ReplaceFn<BaseT>(
        [f = std::forward<FnT>(callback)](BaseT base)
            -> mlir::CyclicAttrTypeReplacer::ReplaceFnResult<BaseT> {
          if constexpr (std::is_same_v<T, BaseT>) {
            FailureOr<BaseT> ret = f(base);
            if (succeeded(ret))
              return {{*ret, WalkResult::skip()}};
            else
              return {{nullptr, WalkResult::interrupt()}};
          }
          if (auto derived = dyn_cast<T>(base)) {
            FailureOr<BaseT> ret = f(derived);
            if (succeeded(ret))
              return {{*ret, WalkResult::skip()}};
            else
              return {{nullptr, WalkResult::interrupt()}};
          }
          return {};
        }));
  }

  template <typename FnT>
  void addCycleBreaker(FnT &&callback) {
    impl.addCycleBreaker(std::forward<FnT>(callback));
  }

  /// Convenience helper for replacing parameters and returning parameters.
  FailureOr<TypedAttr> replaceParameter(TypedAttr attr) {
    FailureOr<Attribute> attrOr = replace(attr);
    if (failed(attrOr))
      return failure();

    return cast<TypedAttr>(*attrOr);
  }

private:
  mlir::CyclicAttrTypeReplacer impl;
};

LogicalResult Lowering::replaceElementsIn(Operation *op, bool replaceAttrs,
                                          bool replaceLocs, bool replaceTypes) {
  // The new element if it changed, null if it did not, so a caller only
  // re-sets what actually moved.
  auto replaceIfDifferent =
      [&](auto element) -> FailureOr<std::conditional_t<
                            std::is_convertible_v<decltype(element), Attribute>,
                            Attribute, Type>> {
    auto replaced = replace(element);
    if (failed(replaced))
      return failure();
    return *replaced != element ? *replaced : nullptr;
  };

  if (replaceAttrs) {
    FailureOr<Attribute> newAttrs =
        replaceIfDifferent(op->getRawDictionaryAttrs());
    if (failed(newAttrs))
      return failure();
    if (*newAttrs)
      op->setDiscardableAttrs(cast<DictionaryAttr>(*newAttrs));

    // Inherent attributes live in properties, outside the dictionary.
    LogicalResult inherent = success();
    if (op->getPropertiesStorageSize())
      op->getName().walkInherentAttrs(op, [&](StringRef, Attribute &attr) {
        if (failed(inherent))
          return;
        FailureOr<Attribute> replaced = replaceIfDifferent(attr);
        if (failed(replaced))
          inherent = failure();
        else if (*replaced)
          attr = *replaced;
      });
    if (failed(inherent))
      return failure();
  }

  if (!replaceTypes && !replaceLocs)
    return success();

  if (replaceLocs) {
    FailureOr<Attribute> newLoc = replaceIfDifferent(op->getLoc());
    if (failed(newLoc))
      return failure();
    if (*newLoc)
      op->setLoc(cast<LocationAttr>(*newLoc));
  }

  if (replaceTypes) {
    for (OpResult result : op->getResults()) {
      FailureOr<Type> newType = replaceIfDifferent(result.getType());
      if (failed(newType))
        return failure();
      if (*newType)
        result.setType(*newType);
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument &arg : block.getArguments()) {
        if (replaceLocs) {
          FailureOr<Attribute> newLoc = replaceIfDifferent(arg.getLoc());
          if (failed(newLoc))
            return failure();
          if (*newLoc)
            arg.setLoc(cast<LocationAttr>(*newLoc));
        }
        if (replaceTypes) {
          FailureOr<Type> newType = replaceIfDifferent(arg.getType());
          if (failed(newType))
            return failure();
          if (*newType)
            arg.setType(*newType);
        }
      }
    }
  }
  return success();
}

/// Contains the two lowerings a LIT module needs, one per domain. One lowers a
/// type in the value domain: what the type is as a parameter value. The other
/// lowers it in the type domain: the layout it occupies, a KGEN storage type.
/// Each has its own rules and its own cache. A type constant carries one answer
/// from each domain: its type value is the value domain's and its MLIR type is
/// the type domain's.
///
/// The value domain calls nothing but itself. Computing a type's value
/// representation never demands a layout, so no cycle can close in the value
/// domain, and every cycle closes in the type domain, where we can break
/// indirect references with pointer types.
///
/// The type domain delegates one thing to the value domain - the value-domain
/// half of a type constant - and then continues into the result itself, because
/// a value-domain result has type-domain positions of its own.
struct LowerLITReplacer {
  Lowering asValue;
  Lowering asType;
};
} // namespace

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

namespace {
/// Evaluation context for LowerLIT that maps LIT struct types to KGEN struct
/// generators via the StructDecls mapping.
class LowerLITEvaluationContext : public SymTabEvaluationContext {
public:
  LowerLITEvaluationContext(ModuleOp module,
                            mlir::LockedSymbolTableCollection &symtab,
                            StructDecls &decls)
      : SymTabEvaluationContext(module, symtab), decls(decls) {}

protected:
  /// Resolve LIT struct types to KGEN struct generators using the decls
  /// mapping.
  FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                  bool acceptAsync) override;

  /// Handle DowncastAttr so that conforms_to expressions with
  /// trait-constrained type parameters can resolve during LowerLIT.
  FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) override;

private:
  StructDecls &decls;
};
} // namespace

static FailureOr<Type> lowerStructType(StructDecls &decls,
                                       LowerLITReplacer &replacer,
                                       ParameterEvaluationContext &evalContext,
                                       MLIRContext *ctx, Type noneType,
                                       LIT::StructType ref) {
  StructDecl &decl = decls.get(ref.getName());
  // Substitute the given parameters in.
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
  evaluator.setEvaluationContext(&evalContext);

  SmallVector<Type> fieldTypes;
  for (Type type : llvm::make_second_range(decl.fields)) {
    if (auto ptrType = dyn_cast<PointerType>(type)) {
      fieldTypes.push_back(PointerType::get(
          noneType, evaluator.getReboundAttribute(ptrType.getAddressSpace()),
          ptrType.getIsNonNull()));
      continue;
    }

    Type reboundType = evaluator.getReboundType(type);
    if (!reboundType)
      return failure();
    fieldTypes.push_back(reboundType);
  }
  if (decl.isSingleElement())
    return replacer.asType.replace(fieldTypes.front());
  // Replace each field type individually, then create the struct.
  SmallVector<Type> replacedTypes;
  replacedTypes.reserve(fieldTypes.size());
  for (Type t : fieldTypes) {
    auto replaced = replacer.asType.replace(t);
    if (failed(replaced) || !*replaced)
      return failure();
    replacedTypes.push_back(*replaced);
  }
  TypedAttr reboundAlignment = evaluator.getReboundAttribute(decl.minAlignment);
  FailureOr<TypedAttr> loweredAlignmentOr =
      replacer.asType.replaceParameter(reboundAlignment);
  if (failed(loweredAlignmentOr))
    return failure();
  // Resolve the parametric isMemoryOnly through the evaluator.
  TypedAttr reboundIsMemoryOnly =
      evaluator.getReboundAttribute(decl.isMemoryOnlyAttr);
  FailureOr<TypedAttr> loweredIsMemoryOnlyOr =
      replacer.asType.replaceParameter(reboundIsMemoryOnly);
  if (failed(loweredIsMemoryOnlyOr))
    return failure();
  return KGEN::StructType::get(ctx, replacedTypes, *loweredIsMemoryOnlyOr,
                               *loweredAlignmentOr);
}

FailureOr<ResolvedStructHandle>
LowerLITEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                           bool /*acceptAsync*/) {
  // LowerLITEvaluationContext does not support async concretization, so
  // acceptAsync is ignored - we always return the generator.

  // We can only resolve if the type reference is a resolved LIT struct type.
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return failure();

  // A constant still in LIT references the struct in both halves; one already
  // lowered references it in neither and resolves through the KGEN generator
  // below.
  auto structType = sugarDynCast<LIT::StructType>(typeParam.getTypeValue());
  if (!structType)
    return SymTabEvaluationContext::resolveStructOp(typeValue, false);

  SymbolRefAttr structDeclRef = structType.getSymbol();
  StringAttr leafName = structDeclRef.getLeafReference();

  auto it = decls.structDecls.find(leafName);
  if (it != decls.structDecls.end()) {
    auto structDecl =
        symtab.lookupSymbolIn<StructGeneratorOp>(module, it->second.symRef);
    if (!structDecl)
      return failure();
    return ResolvedStructHandle{
        cast<StructDeclInterface>(structDecl.getOperation()),
        structType.getParamValues(), nullptr,
        /*instance=*/nullptr};
  }

  return SymTabEvaluationContext::resolveStructOp(typeValue, false);
}

FailureOr<TypedAttr> LowerLITEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Fold DowncastAttr when the input is a concrete struct type value. Unwrap
  // and expose the concrete type value, which can be used to further simplify
  // things like conforms_to.
  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr))
    if (TypedAttr folded = LIT::foldDowncastToStructType(downcast))
      return folded;

  return SymTabEvaluationContext::evaluateContextSpecific(attr);
}

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

/// Populate `replacer` with the lowering patterns for attributes and types
/// from the computed lowerings for each struct decl.
static void populateTypeDomainRules(StructDecls &decls,
                                    LowerLITReplacer &replacer,
                                    ParameterEvaluationContext &evalContext,
                                    MLIRContext *ctx) {
  auto typeType = TypeType::get(ctx);
  auto emptyStructType = KGEN::StructType::get(ctx, ArrayRef<Type>{});
  auto emptyStruct = StructAttr::get({}, emptyStructType);
  auto noneType = KGEN::NoneType::get(ctx);

  replacer.asType.addRule(
      [&replacer, evalCtxPtr = &evalContext](
          BindParamsAttr bindParams) -> FailureOr<Attribute> {
        // We always simplify BindParamsAttr against a evaluation context.
        SmallVector<TypedAttr> loweredParams;
        for (TypedAttr param : bindParams.getParamValues()) {
          auto replaced = replacer.asType.replace(param);
          if (failed(replaced))
            return failure();
          loweredParams.push_back(cast<TypedAttr>(*replaced));
        }
        auto generatorOr = replacer.asType.replace(bindParams.getGenerator());
        if (failed(generatorOr))
          return failure();

        // BindParamsAttr has to be constructed with an evaluation context to
        // fold properly.
        TypedAttr evaluated = BindParamsAttr::get(
            bindParams.getContext(), cast<TypedAttr>(*generatorOr),
            loweredParams, bindParams.getDischarged(), evalCtxPtr);
        return evaluated;
      });

  // A type constant is the one place a value-domain position occurs inside a
  // type-domain tree. Its type value goes to the value domain first, and the
  // result is then lowered here as well, because a value-domain result has
  // type-domain positions of its own.
  replacer.asType.addRule(
      [&replacer](TypeParamAttr typeValue) -> FailureOr<Attribute> {
        FailureOr<Type> valueHalfOr =
            replacer.asValue.replace(typeValue.getTypeValue());
        if (failed(valueHalfOr))
          return failure();
        auto typeValueOr = replacer.asType.replace(*valueHalfOr);
        auto mlirTypeOr = replacer.asType.replace(typeValue.getMlirType());
        auto typeOr = replacer.asType.replace(typeValue.getType());
        if (failed(typeValueOr) || failed(mlirTypeOr) || failed(typeOr))
          return failure();

        return TypeParamAttr::get(*typeValueOr, *mlirTypeOr, *typeOr);
      });

  // NOTE: Downcast becomes an no-op after lower-lit. However, we should
  // probably keep the attr till elaboration time after we preserve traits
  // properly in KGEN for a better error message.
  // We simply strip all downcast at the moment otherwise all downcasts will be
  // in same (useless) form of `#downcast<T> : !kgen.type` anyway.
  //
  // TODO: preserve trait symbol in KGEN for downcast/conforms_to/is_sub_trait.
  replacer.asType.addRule(
      [&replacer](DowncastAttr downcast) -> FailureOr<Attribute> {
        auto typeOr = replacer.asType.replace(downcast.getType());
        if (failed(typeOr))
          return failure();
        auto downcastValOr =
            replacer.asType.replaceParameter(downcast.getInputTypeValue());
        if (failed(downcastValOr))
          return failure();
        // Since we are erasing the trait target type, the downcast becomes
        // essentially an upcast to type.type
        return UpcastAttr::get(*typeOr, *downcastValOr);
      });
  replacer.asType.addRule(
      [](IsRefinedTypeAttr isRefinedTrait) -> FailureOr<Attribute> {
        return SIMDAttr::getScalarBool(isRefinedTrait.getContext(), true);
      });

  // All metatypes lower to `!kgen.type`.
  replacer.asType.addRule([=](StructMetaType) { return typeType; });
  replacer.asType.addRule([=](StructMetaMetaType) { return typeType; });
  replacer.asType.addRule([=](AnyTraitType) { return typeType; });
  replacer.asType.addRule(
      [=](FnLiteralTypeGeneratorMetaType) { return typeType; });
  replacer.asType.addRule(
      [=](FnLiteralTypeGeneratorMetaMetaType) { return typeType; });
  replacer.asType.addRule([=](NonStructTypeType) { return typeType; });

  // #lit.ref.pack => #kgen.struct
  replacer.asType.addRule(
      [&replacer](RefPackAttr refPack) -> FailureOr<Attribute> {
        SmallVector<TypedAttr> loweredElts;
        loweredElts.reserve(refPack.getValues().size());
        for (TypedAttr elt : refPack.getValues()) {
          auto eltOr = replacer.asType.replaceParameter(elt);
          if (failed(eltOr))
            return failure();
          loweredElts.push_back(*eltOr);
        }
        FailureOr<Type> typeOr = replacer.asType.replace(refPack.getType());
        if (failed(typeOr))
          return failure();
        return StructAttr::get(loweredElts, cast<KGEN::StructType>(*typeOr));
      });

  // !lit.ref.pack<:param_list<!kgen.type> types, owned_in_mem, mut origin, 42>
  // => !kgen.struct<variadic_ptr_map(types), 42>
  replacer.asType.addRule([&replacer](RefPackType ref) -> FailureOr<Type> {
    auto variadicOr = replacer.asType.replaceParameter(ref.getVariadic());
    auto addrSpaceOr = replacer.asType.replaceParameter(ref.getAddressSpace());
    if (failed(variadicOr) || failed(addrSpaceOr))
      return failure();
    auto mapped =
        ParamOperatorAttr::get(POC::VariadicPtrMap, *variadicOr, *addrSpaceOr);
    return KGEN::StructType::get(ref.getContext(), mapped,
                                 /*memOnly=*/false, /*minAlign*/ {},
                                 /*isParamPack=*/true);
  });

  // !lit.ref -> !kgen.pointer
  replacer.asType.addRule([&replacer](RefType ref) -> FailureOr<Type> {
    auto elemTpOr = replacer.asType.replace(ref.getElementType());
    auto addrSpaceOr = replacer.asType.replaceParameter(ref.getAddressSpace());
    if (failed(elemTpOr) || failed(addrSpaceOr))
      return failure();
    return PointerType::get(*elemTpOr, *addrSpaceOr);
  });

  // Replace all origin attributes with empty structs. These attributes are
  // all terminal.
  replacer.asType.addRule([=](AnyOriginAttr) { return emptyStruct; });
  replacer.asType.addRule([=](StaticOriginAttr) { return emptyStruct; });
  replacer.asType.addRule([=](ComptimeOriginAttr) { return emptyStruct; });
  replacer.asType.addRule([=](OriginUnionAttr) { return emptyStruct; });
  replacer.asType.addRule([=](OriginMutCastAttr) { return emptyStruct; });
  replacer.asType.addRule([=](ImplicitOriginRefAttr) { return emptyStruct; });
  replacer.asType.addRule([=](OriginSetAttr) { return emptyStruct; });
  replacer.asType.addRule([=](EllipsisAttr) { return emptyStruct; });
  replacer.asType.addRule([](OriginEqAttr) -> FailureOr<Attribute> {
    llvm_unreachable("OriginEqAttr should be replaced by now");
  });

  // !lit.origin -> !kgen.struct<()>
  replacer.asType.addRule([=](OriginType) { return emptyStructType; });
  replacer.asType.addRule([=](EllipsisType) { return emptyStructType; });
  replacer.asType.addRule([=](OriginSetType) { return emptyStructType; });

  // A struct whose layout names itself - for example through a function-typed
  // field whose signature takes the struct - has no finite structural layout,
  // so the recursion is cut at the re-entry with an opaque pointer. The field
  // must stay a function type, but its signature is a placeholder: the
  // interpreter re-presents the stored symbol with `#kgen.func_ptr_bitcast`,
  // and each use of the field casts to a type lowered outside the cycle.
  replacer.asType.addCycleBreaker([noneType](Type t) -> std::optional<Type> {
    if (!isa<LIT::StructType>(t))
      return std::nullopt;
    return Type(PointerType::get(noneType));
  });

  // #lit.struct -> #kgen.struct
  replacer.asType.addRule(
      [&, noneType](LITStructAttr attr) -> FailureOr<Attribute> {
        LIT::StructType ref = attr.getType();
        StructDecl &decl = decls.get(ref.getName());

        SmallVector<TypedAttr> values;
        values.reserve(attr.getValues().size());
        for (auto [entry, type] : llvm::zip(attr.getValues(), decl.fields)) {
          FailureOr<TypedAttr> valueOr =
              replacer.asType.replaceParameter(std::get<1>(entry));
          if (failed(valueOr))
            return failure();

          TypedAttr value = *valueOr;
          // We to check if this is a value for a struct field that is known to
          // be a pointer type, in which case we erase the element type
          // but preserve other pointer attributes like nonnull.
          if (isa<PointerType>(type.second)) {
            auto type = cast<PointerType>(value.getType());
            auto ptrType = PointerType::get(noneType, type.getAddressSpace(),
                                            type.getIsNonNull());
            value = ParamOperatorAttr::get(POC::PtrBitcast, value, ptrType);
          }
          values.push_back(value);
        }

        if (decl.isSingleElement())
          return values.front();

        auto refOr = replacer.asType.replace(ref);
        if (failed(refOr))
          return failure();
        if (auto type = cast_or_null<KGEN::StructType>(*refOr))
          return StructAttr::get(values, type);
        return failure();
      });

  // #lit.struct.extract -> #kgen.struct.extract
  replacer.asType.addRule(
      [&](LIT::StructExtractAttr attr) -> FailureOr<Attribute> {
        auto ref = cast<LIT::StructType>(attr.getStructValue().getType());
        int idx = decls.fieldIndices.at({ref.getName(), attr.getField()});
        auto valueOr = replacer.asType.replaceParameter(attr.getStructValue());
        if (failed(valueOr))
          return failure();
        if (decls.get(ref.getName()).isSingleElement())
          return *valueOr;
        return KGEN::StructExtractAttr::get(*valueOr, idx);
      });

  // Sugar attr is turned into canonical form.
  replacer.asType.addRule([&](SugarAttr sugar) {
    llvm_unreachable("sugar should be replaced by now");
    return Attribute();
  });

  // A trait's layout is `!kgen.type`; its value-domain form is registered in
  // `populateValueDomainRules`.
  replacer.asType.addRule(
      [=](TraitType) -> FailureOr<Type> { return typeType; });

  replacer.asType.addRule([&](TraitSymbolAttr attr) -> FailureOr<Attribute> {
    if (attr.getSymbol().getLeafReference().strref().ends_with(
            UNI_CLOSURE_TRAIT_NAME)) {
      SmallVector<TypedAttr> params;
      for (auto param : attr.getParamValues()) {
        // Pog lists are irrelevant after lower-lit.
        if (isa<PogListAttr>(QuoteAttr::unquote(param)))
          continue;
        auto replaced = replacer.asType.replaceParameter(param);
        if (failed(replaced))
          return failure();

        // strip the origin metadata too.
        if (auto meta = dyn_cast<FnMetadataAttr>(*replaced))
          params.push_back(meta.getWithMetadata(nullptr));
        else
          params.push_back(*replaced);
      }
      // After pog list is skipped, there are 4 parameter remaining.
      assert(params.size() == 4);
      return TraitSymbolAttr::get(attr.getSymbol(), params);
    }

    // It has to be a non-parametric trait for non-closures.
    assert(attr.getParamValues().empty());
    return attr;
  });

  // Since lowerings have been generated for all struct types, we just need to
  // lookup the lowered type and substitute the parameters.
  replacer.asType.addRule([&, ctx,
                           noneType](LIT::StructType ref) -> FailureOr<Type> {
    return lowerStructType(decls, replacer, evalContext, ctx, noneType, ref);
  });
}

/// Register the value domain rules.
///
/// Lowering in the value domain is a dead-end; it must never break out back
/// into the type domain. If computing the value representation of a type ever
/// recursed into demanding the layout of its subtypes, we could hit a cycle.
static void populateValueDomainRules(StructDecls &decls,
                                     LowerLITReplacer &replacer,
                                     MLIRContext *ctx) {
  auto typeType = TypeType::get(ctx);

  // Registered first so every type rule below takes precedence.
  replacer.asValue.addRule(
      [](Attribute attr) -> FailureOr<Attribute> { return attr; });

  // A metatype has the one representation in both domains, `!kgen.type`, and
  // nothing under it is worth descending into.
  replacer.asValue.addRule(
      [=](StructMetaType) -> FailureOr<Type> { return typeType; });

  // A generator's parameter decl types are always types; only its body is a
  // value. The type domain reaches them by descending into this result and
  // lowers them there.
  replacer.asValue.addRule([&replacer](GeneratorType gen) -> FailureOr<Type> {
    auto bodyOr = replacer.asValue.replace(gen.getBody());
    if (failed(bodyOr))
      return failure();
    return GeneratorType::get(gen.getInputParamTypes(), *bodyOr,
                              gen.getParamListAttrs());
  });

  replacer.asValue.addRule([](ParamType paramRef) -> FailureOr<Type> {
    return TypeValueType::get(paramRef.getParam());
  });

  replacer.asValue.addRule([=](TraitType traitType) -> FailureOr<Type> {
    return TypeValueType::get(
        TraitInstanceRefAttr::get(ctx, traitType.getSymbols(), typeType));
  });

  // A struct reference becomes a generator reference over the same parameter
  // values. Its metatype is `!kgen.type`, like every metatype.
  replacer.asValue.addRule([&decls,
                            typeType](LIT::StructType ref) -> FailureOr<Type> {
    StringAttr leafName = ref.getValue().getValue().getLeafReference();
    StructDecl &decl = decls.structDecls.find(leafName)->second;
    return TypeValueType::get(
        TypeGeneratorRefAttr::get(decl.symRef, ref.getParamValues(), typeType));
  });
}

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

namespace {
/// Struct operations need to refer to the struct declaration symbol.
struct LITTypeLowerer : public IRRewriter, LowerLITReplacer {
  explicit LITTypeLowerer(ModuleOp module, StructDecls &structDecls,
                          mlir::LockedSymbolTableCollection &symtab);

  /// Get the index of the struct field.
  int getField(StringAttr name, LIT::StructType ref) {
    return structDecls.fieldIndices.lookup({ref.getName(), name});
  }
  /// Return true if the struct is single element.
  bool isSingleElement(LIT::StructType ref) {
    return structDecls.get(ref.getName()).isSingleElement();
  }
  Value getCastedToType(Location loc, Value value, Type type);

  /// Materialize destination conversions.
  template <typename OpT>
  LogicalResult materializeLowering(OpT op);

  /// Evaluation context used for simplifying parameters.
  LowerLITEvaluationContext evalContext;
  /// The struct decl map.
  StructDecls &structDecls;
  /// Converter for debuginfo.
  DebugInfo::DebugInfoNonCyclicTypeConverter debugTypeConverter;
  /// Unrealized casts to resolve at the end of type lowering.
  SmallVector<mlir::UnrealizedConversionCastOp> unrealizedCasts;
};
} // namespace

static DebugInfo::DIType buildDebugInfoForStructRef(
    LIT::StructType ref, StructDecls &structDecls,
    DebugInfo::DebugInfoNonCyclicTypeConverter &converter,
    ParameterEvaluationContext &evalContext) {
  // Substitute parameters into the field types.
  StructDecl &decl = structDecls.get(ref.getName());
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
  evaluator.setEvaluationContext(&evalContext);

  auto getDebugInfoType = [&](const std::pair<StringAttr, Type> &nameAndType) {
    auto [name, type] = nameAndType;
    auto reboundType = evaluator.getReboundType(type);
    DebugInfo::DIType fieldDIType = converter.convertDebugType(reboundType);
    if (!fieldDIType) {
      fieldDIType = converter.convertDebugType(
          PointerType::get(KGEN::NoneType::get(type.getContext())));
    }
    return DebugInfo::DIMemberType::get(name, fieldDIType);
  };

  // Flatten register-passable, single-element structs.
  // TODO(#23914): Track this optimization with DWARF expressions.
  if (decl.fields.size() == 1 && decl.isRegisterPassable())
    return getDebugInfoType(decl.fields.front()).getType();

  SmallVector<DebugInfo::DIMemberType> elementTypes =
      llvm::map_to_vector(decl.fields, getDebugInfoType);

  // Parameterize the raw source name.
  DebugInfo::SourceNameAttr sourceName = decl.sourceName;
  // TODO: Make StructDeclOp's sourceName a DefaultValuedAttr once properties
  // play nicely with it.
  if (!sourceName) {
    std::string name;
    llvm::raw_string_ostream os(name);
    printNestedSymbolReference(os, ref.getSymbol());
    sourceName =
        DebugInfo::SourceNameAttr::get(StringAttr::get(ref.getContext(), name),
                                       DebugInfo::SourceNameKind::Struct);
  }

  SmallVector<StringAttr> paramValues;
  for (TypedAttr value : ref.getParamValues())
    paramValues.push_back(getParamTypeAsString(value));
  sourceName = DebugInfo::SourceNameAttr::get(
      sourceName.getName(), sourceName.getParamTypes(),
      sourceName.getArgTypes(), paramValues, sourceName.getParent(),
      sourceName.getKind(), sourceName.getDecorators());

  return DebugInfo::DIStructType::get(sourceName.encode(), elementTypes);
}

LITTypeLowerer::LITTypeLowerer(ModuleOp module, StructDecls &structDecls,
                               mlir::LockedSymbolTableCollection &symtab)
    : IRRewriter(module.getContext()), evalContext(module, symtab, structDecls),
      structDecls(structDecls) {
  populateValueDomainRules(structDecls, *this, module.getContext());
  populateTypeDomainRules(structDecls, *this, evalContext, module.getContext());

  // Build a converter to handle updating converted types within debug info
  // constructs.
  debugTypeConverter.addConversion([&](Type type) -> std::optional<Type> {
    FailureOr<Type> newTypeOr = asType.replace(type);
    if (succeeded(newTypeOr) && *newTypeOr != type)
      return debugTypeConverter.convertDebugType(*newTypeOr);
    return std::nullopt;
  });
  debugTypeConverter.addConversion(
      [&](LIT::StructType type) -> DebugInfo::DIType {
        return buildDebugInfoForStructRef(type, structDecls, debugTypeConverter,
                                          evalContext);
      });
  debugTypeConverter.addConversion([&](PointerType type) -> DebugInfo::DIType {
    DebugInfo::DIType elementType =
        debugTypeConverter.convertDebugType(type.getElementType());
    if (!elementType) {
      // If the type that we point to can't be converted into a
      // debuginfo type, make a None pointer debuginfo type.
      elementType = debugTypeConverter.convertDebugType(
          KGEN::NoneType::get(type.getContext()));
    }
    return DebugInfo::DITargetIndependentPointerType::get(elementType);
  });
  debugTypeConverter.addConversion([&](RefType type) -> DebugInfo::DIType {
    return debugTypeConverter.convertDebugType(type.getAsPointerType());
  });

  // Debug info describes storage, so converting it belongs to the type domain.
  // The converter lowers the types it meets through `asType`, which is why it
  // must not be reachable from `asValue`.
  asType.addRule([&](DebugInfo::DIType type) {
    return debugTypeConverter.convertDebugType(type);
  });
}

static Value lowerOp(StructInsertOp op, StructInsertOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  LIT::StructType ref = op.getContainer().getType();
  if (b.isSingleElement(ref))
    return adaptor.getValue();

  int index = b.getField(op.getFieldAttr(), ref);
  return StructReplaceOp::create(b, op.getLoc(), adaptor.getValue(),
                                 adaptor.getContainer(), index);
}

static Value lowerOp(LIT::StructExtractOp op,
                     LIT::StructExtractOpAdaptor adaptor, LITTypeLowerer &b) {
  LIT::StructType ref = op.getContainer().getType();
  if (b.isSingleElement(ref))
    return adaptor.getContainer();

  int index = b.getField(op.getFieldAttr(), ref);
  return KGEN::StructExtractOp::create(b, op.getLoc(), adaptor.getContainer(),
                                       b.getIndexAttr(index));
}

static TypedAttr getAlignmentFromType(Type type, LITTypeLowerer &b) {
  auto structType = dyn_cast<LIT::StructType>(type);
  if (!structType)
    return {};

  StructDecl &decl = b.structDecls.get(structType.getName());
  if (!decl.minAlignment)
    return {};

  // Substitute the alignment with struct parameters.
  ParameterEvaluator evaluator(decl.decls, structType.getParamValues());
  evaluator.setEvaluationContext(&b.evalContext);
  return evaluator.getReboundAttribute(decl.minAlignment);
}

static Value lowerOp(VarDeclOp op, VarDeclOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  // Lower a lit.var.decl to pop.stack_allocation.
  // Check if the element type is a struct with explicit alignment.
  TypedAttr alignment = getAlignmentFromType(op.getType().getElementType(), b);
  return POP::StackAllocationOp::create(
      b, op.getLoc(), op.getType().getAsPointerType(),
      /*count=*/1, alignment, /*markedLifetimes=*/true);
}

static Value lowerOp(VarLifetimeStartOp op, VarLifetimeStartOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  b.replaceOpWithNewOp<POP::StackAllocLifetimeStartOp>(
      op, op.getArg().getDefiningOp()->getOperand(0));
  return {};
}

static Value lowerOp(VarLifetimeEndOp op, VarLifetimeEndOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  b.replaceOpWithNewOp<POP::StackAllocLifetimeEndOp>(
      op, op.getArg().getDefiningOp()->getOperand(0));
  return {};
}

static Value lowerOp(RefImmutOp op, RefImmutOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefUpcastOp op, RefUpcastOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefToPointerOp op, RefToPointerOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefFromPointerOp op, RefFromPointerOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getPtr();
}

static Value lowerOp(RefFromPointerREPLOp op,
                     RefFromPointerREPLOpAdaptor adaptor, LITTypeLowerer &b) {
  return adaptor.getPtr();
}

static Value lowerOp(RefToKgenPtrOp op, RefToKgenPtrOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefFromKgenPtrOp op, RefFromKgenPtrOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getPointer();
}

static Value lowerOp(RefLoadOp op, RefLoadOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return POP::LoadOp::create(b, op.getLoc(), adaptor.getRef());
}

static Value lowerOp(MaterializeIntoOp op, MaterializeIntoOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  Value dynVal =
      ParamMaterializeOp::create(b, op->getLoc(), adaptor.getValue());
  b.replaceOpWithNewOp<POP::StoreOp>(op, dynVal, adaptor.getDest());
  return {};
}

static Value lowerOp(RefStoreOp op, RefStoreOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  b.replaceOpWithNewOp<POP::StoreOp>(op, adaptor.getValue(), adaptor.getDest());
  return {};
}

static Value lowerOp(MemcpyOp op, MemcpyOpAdaptor adaptor, LITTypeLowerer &b) {
  TypedAttr target =
      ParamOperatorAttr::get(POC::CurrentTarget, {}, b.getType<TargetType>());
  Value dst = adaptor.getDst();
  auto elementTypeAttr =
      TypeParamAttr::get(cast<PointerType>(dst.getType()).getElementType(),
                         TypeType::get(dst.getContext()));
  Value sizeOfElt = ParamConstantOp::create(
      b, op.getLoc(),
      ParamOperatorAttr::get(POC::GetSizeOf, {elementTypeAttr, target}));
  // Note - swap the source and destination operands around.
  b.replaceOpWithNewOp<POP::MemcpyOp>(op, dst, adaptor.getSrc(), sizeOfElt);
  return {};
}

static Value lowerOp(RefStructGEROp op, RefStructGEROpAdaptor adaptor,
                     LITTypeLowerer &b) {
  if (op.usesFieldAccess()) {
    // Field name access: lower to kgen.struct.gep with computed index
    auto ref =
        cast<LIT::StructType>(op.getContainer().getType().getElementType());
    if (b.isSingleElement(ref))
      return adaptor.getContainer();

    int index = b.getField(op.getFieldAttr(), ref);
    return StructGEPOp::create(b, op.getLoc(), adaptor.getContainer(), index);
  } else {
    // Index access: lower to kgen.struct.gep with parametric index

    // Check if this is a single-element struct, similar to field access.
    // For single-element structs, the container IS the element, so just
    // return it directly instead of creating a GEP.
    auto elementType = op.getContainer().getType().getElementType();
    if (auto ref = dyn_cast<LIT::StructType>(elementType)) {
      if (b.isSingleElement(ref))
        return adaptor.getContainer();
    }

    auto resultTypeOr = b.asType.replace(op.getType());
    if (failed(resultTypeOr))
      return nullptr;
    auto resultType = cast<PointerType>(*resultTypeOr);

    // Handle single-element struct flattening for parametric types.
    // When a single-element trivial struct is accessed through parametric
    // types (e.g., trait Self type), the struct may be flattened during
    // lowering. In either case, there's no struct to GEP into, so return
    // the container directly. We must check for ParamType to avoid
    // prematurely short-circuiting parametric cases that will resolve to
    // multi-element structs.
    auto containerPtrType = cast<PointerType>(adaptor.getContainer().getType());
    Type containerElemType = containerPtrType.getElementType();

    // Detection method 1: Types match after lowering (identity operation).
    // This catches cases where parametric types have resolved identically.
    bool isIdentity = adaptor.getContainer().getType() == resultType;

    auto isNonPackStruct =
        isa<KGEN::StructType>(containerElemType) &&
        !cast<KGEN::StructType>(containerElemType).getIsParamPack();

    // Detection method 2: Container already flattened to a concrete scalar.
    // This catches cases where the struct was flattened before this point.
    bool isFlattenedNonStruct =
        !isNonPackStruct && !isa<KGEN::ParamType>(containerElemType);

    if (isIdentity || isFlattenedNonStruct)
      return adaptor.getContainer();

    return StructGEPOp::create(b, op.getLoc(), resultType,
                               adaptor.getContainer(), *op.getIndex());
  }
}

/// Squash noop rebinds exposed by ref -> ptr lowering.
static Value lowerOp(RebindOp op, RebindOpAdaptor adaptor, LITTypeLowerer &b) {
  // If this is a noop after lowering, squish it
  if (adaptor.getInput().getType() == b.asType.replace(op.getType()))
    return adaptor.getInput();
  // Otherwise just leave it and type replacement will form a valid rebind
  // in the new type domain.
  return op.getResult();
}

// lit.ref.pack.create => kgen.struct.create
static Value lowerOp(RefPackCreateOp op, RefPackCreateOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  auto typeOr = b.asType.replace(op.getType());
  if (failed(typeOr))
    return nullptr;
  return StructCreateOp::create(b, op.getLoc(), *typeOr, adaptor.getOperands());
}

// lit.ref.pack.extract => kgen.struct.extract
static Value lowerOp(RefPackExtractOp op, RefPackExtractOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  Value value = KGEN::StructExtractOp::create(b, op.getLoc(), adaptor.getPack(),
                                              adaptor.getIndex());
  // If the result didn't fold to a pointer type, we need to emit a rebind.
  FailureOr<Type> expectedOr = b.asType.replace(op.getType());
  if (failed(expectedOr))
    return nullptr;
  if (value.getType() != *expectedOr)
    value = RebindOp::create(b, op.getLoc(), *expectedOr, value);
  return value;
}

static Value lowerOp(RefPackFromPointerPackOp op,
                     RefPackFromPointerPackOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getPack();
}

static Value lowerVersionOp(Operation *op, int64_t number, LITTypeLowerer &b) {
  return ParamConstantOp::create(
      b, op->getLoc(),
      KGEN::SIMDAttr::get(
          KGEN::DTypeValue(number, KGENDType::index),
          SIMDType::get(
              /*size=*/1,
              DTypeConstantAttr::get(op->getContext(), KGENDType::index))));
}

// lit.mojo.version.major => kgen.param.constant : scalar<index>
static Value lowerOp(MojoVersionMajorOp op, MojoVersionMajorOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return lowerVersionOp(op, M::getMojoVersion().major, b);
}

// lit.mojo.version.minor => kgen.param.constant : scalar<index>
static Value lowerOp(MojoVersionMinorOp op, MojoVersionMinorOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return lowerVersionOp(op, M::getMojoVersion().minor, b);
}

// lit.mojo.version.patch => kgen.param.constant : scalar<index>
static Value lowerOp(MojoVersionPatchOp op, MojoVersionPatchOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return lowerVersionOp(op, M::getMojoVersion().patch, b);
}

Value LITTypeLowerer::getCastedToType(Location loc, Value value, Type type) {
  // If already casted, done.
  if (value.getType() == type)
    return value;

  // If coming from a cast, use input.
  if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    return getCastedToType(loc, castOp.getOperand(0), type);

  // Otherwise create a new cast.
  auto cast = mlir::UnrealizedConversionCastOp::create(*this, loc, type, value);
  unrealizedCasts.push_back(cast);
  return cast.getResult(0);
}

template <typename OpT>
LogicalResult LITTypeLowerer::materializeLowering(OpT op) {
  setInsertionPoint(op);
  SmallVector<Value> castedOperands;
  castedOperands.reserve(op->getNumOperands());
  // Get type adjusted values into the adaptor to simplify clients.
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();

    auto newTypeOr = asType.replace(value.getType());
    if (failed(newTypeOr))
      return failure();
    // When value is a function argument, location info's function scope is
    // different from the operations in the function body. Use op->getLoc()
    // for new cast op's location instead of using value.loc().
    castedOperands.push_back(getCastedToType(op->getLoc(), value, *newTypeOr));
  }

  typename OpT::Adaptor adaptor(castedOperands, op->getAttrDictionary(),
                                op.getProperties());
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    Value result = lowerOp(op, adaptor, *this);
    if (!result)
      return failure();
    if (result.getType() != resultType)
      result = getCastedToType(result.getLoc(), result, resultType);

    if (op->getResult(0) != result)
      replaceOp(op, {result});
  } else {
    assert(op->getNumResults() == 0);
    [[maybe_unused]] Value result = lowerOp(op, adaptor, *this);
    assert(!result && "nullary lowering shouldn't produce an op");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entrypoint.
//===----------------------------------------------------------------------===//

LogicalResult LIT::lowerLITTypes(ModuleOp module, StructDecls &state,
                                 mlir::LockedSymbolTableCollection &symtab) {
  // Reject the layouts that contain themselves by value.
  if (failed(analyzeStructRefs(state)))
    return failure();
  LITTypeLowerer b(module, state, symtab);

  // Lower operations first.
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<MaterializeIntoOp, StructInsertOp, StructExtractOp, RefImmutOp,
              RefUpcastOp, RefToPointerOp, RefFromPointerOp,
              RefFromPointerREPLOp, RefToKgenPtrOp, RefFromKgenPtrOp,
              RefStructGEROp, RefLoadOp, RefStoreOp, MemcpyOp, RebindOp,
              RefPackCreateOp, RefPackExtractOp, RefPackFromPointerPackOp,
              VarDeclOp, VarLifetimeStartOp, VarLifetimeEndOp,
              MojoVersionMajorOp, MojoVersionMinorOp, MojoVersionPatchOp>(
            [&](auto op) { return b.materializeLowering(op); })
        .Default([&](auto op) { return success(); });
  });
  if (result.wasInterrupted())
    return failure();

  // FIXME(MOCO-4167): Duplicate a kgen-lowered witness entry, during lower-lit,
  // we might have ordering issue during lit->kgen conversion, depending on
  // whether `get_witness_attr` is evaluated before/after the referenced struct
  // generator is lowered. It might or might not be folded correctly.
  //
  // Simply postpone the struct generator lowering to the last step (as we are
  // already doing) won't help either, as the witness_op might also have a
  // witness_attr inside for complicated cases.
  for (StructGeneratorOp structGen : module.getOps<StructGeneratorOp>()) {
    for (auto conformsOp : structGen.getOps<ConformanceOp>()) {
      for (WitnessOp witnessOp : conformsOp.getOps<WitnessOp>()) {
        b.setInsertionPoint(witnessOp);
        Operation *kgenWitnessOp = b.clone(*witnessOp);
        witnessOp.setSymName(std::string(witnessOp.getSymName()) + ".#lit#");
        LogicalResult res = b.asType.replaceElementsIn(kgenWitnessOp,
                                                       /*replaceAttrs=*/true,
                                                       /*replaceLocs=*/true,
                                                       /*replaceTypes=*/true);
        if (failed(res))
          return failure();
      }
    }
  }

  result = module.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Skip StructGeneratorOps and lower them the last.
    if (auto structGen = dyn_cast<StructGeneratorOp>(op))
      return WalkResult::skip();

    LogicalResult res = b.asType.replaceElementsIn(op, /*replaceAttrs=*/true,
                                                   /*replaceLocs=*/true,
                                                   /*replaceTypes=*/true);

    if (failed(res))
      return WalkResult::interrupt();

    if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      b.setInsertionPoint(cast);
      Type inType = cast.getOperand(0).getType();
      Type outType = cast.getResult(0).getType();
      if (inType == outType) {
        b.replaceOp(cast, cast.getOperand(0));
        return WalkResult::skip();
      } else if (isa<PointerType>(inType) && isa<PointerType>(outType)) {
        b.replaceOpWithNewOp<POP::PointerBitcastOp>(cast, outType,
                                                    cast.getOperand(0));
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });

  // Lower types in StructGeneratorOps last because we need signature and
  // witness entries to keep using LIT types in order for ParameterEvaluator to
  // work smoothly.
  for (StructGeneratorOp structGen : module.getOps<StructGeneratorOp>()) {
    LogicalResult res = b.asType.replaceElementsIn(structGen,
                                                   /*replaceAttrs=*/true,
                                                   /*replaceLocs=*/true,
                                                   /*replaceTypes=*/true);
    if (failed(res))
      return failure();

    // Then lower the body.
    structGen.getBody().walk([&](Operation *op) {
      LogicalResult res = b.asType.replaceElementsIn(op, /*replaceAttrs=*/true,
                                                     /*replaceLocs=*/true,
                                                     /*replaceTypes=*/true);
      if (failed(res))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
  }

  // FIXME(MOCO-4167): Erase the duplicated witness entries at the end after
  // everything is lowered properly.
  for (StructGeneratorOp structGen : module.getOps<StructGeneratorOp>())
    for (auto conformsOp : structGen.getOps<ConformanceOp>())
      for (auto witnessOp :
           llvm::make_early_inc_range(conformsOp.getOps<WitnessOp>()))
        if (witnessOp.getSymName().ends_with(".#lit#"))
          witnessOp.erase();

  if (result.wasInterrupted())
    return failure();

  return success();
}
