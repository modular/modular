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

#include "SingletonTypeHelper.h"

#include "ConcreteBindings.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>

#include "Config/Version.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLIT
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// This processes a `lit.fn` and returns the param declarations for the
/// normal input parameters, ignoring the origin parameters.
static ArrayRef<ParamDeclAttr> extractImplicitOriginParams(FnOp func) {
  size_t numImplicitOrigins =
      func.getFuncTypeGenerator().getNumImplicitOriginDecls();
  return func.getInputParams().drop_back(numImplicitOrigins);
}

/// The param decl positions that have been dropped.
using ParamDeclDropMask = llvm::BitVector;

/// Check a list of parameter declarations to see if any of the parameters are
/// singletons like origin parameters.  If so, remove them from the list.
static ParamDeclDropMask
removeSingletonParamDecls(SingletonTypeHelper &singletonTypeHelper,
                          SmallVectorImpl<ParamDeclAttr> &paramDecls) {
  ParamDeclDropMask mask(paramDecls.size());
  size_t numRemoved = 0;
  for (auto [idx, paramDecl] : llvm::enumerate(paramDecls)) {
    // If this is a parameter we are supposed to remove, bind it.
    if (singletonTypeHelper.isSingletonType(paramDecl.getType())) {
      // We can just remove the parameter without inserting a placeholder
      // in the body. This is safe because we unconditionally replace
      // all attributes of origin type at the end of this pass with
      // #lit.any.origin, which will conveniently get all references to
      // this. That said, we need to remember the index so we can update
      // the signature.
      ++numRemoved;
      mask.set(idx);
      continue;
    }

    // If we removed any before it, copy this down.
    if (numRemoved)
      paramDecls[idx - numRemoved] = paramDecls[idx];
  }

  // Drop any removed parameters.
  paramDecls.resize(paramDecls.size() - numRemoved);
  return mask;
}

/// Lower `lit.bind_params` to a no-op by replacing it with its generator
/// operand. This is only valid when every input parameter on the generator is
/// a singleton type, since those parameters are removed during lowering and
/// the bound/unbound generator types become identical.
static LogicalResult
lowerBindParamsOp(BindParamsOp op, SingletonTypeHelper &singletonTypeHelper) {
  FnTypeGeneratorType genType = op.getGenerator().getType();
  for (auto [idx, paramType] : llvm::enumerate(genType.getInputParamTypes())) {
    if (singletonTypeHelper.isSingletonType(paramType))
      continue;
    return op.emitError()
           << "may only bind singleton compile-time parameters during "
              "lowering; parameter "
           << idx << " has non-singleton type " << paramType;
  }

  IRRewriter rewriter{OpBuilder(op)};
  rewriter.replaceOp(op, op.getGenerator());
  return success();
}

//===----------------------------------------------------------------------===//
// Op Lowering
//===----------------------------------------------------------------------===//

namespace {
struct LITLowerer {
  LITLowerer(mlir::SymbolTableAnalysis &symbolTables,
             DenseMap<StringAttr, StringAttr> &renamedSymbols,
             SingletonTypeHelper &singletonTypeHelper, StructDecls &structDecls)
      : symbolTables(symbolTables), renamedSymbols(renamedSymbols),
        singletonTypeHelper(singletonTypeHelper), structDecls(structDecls),
        typeType(TypeType::get(
            symbolTables.getTopLevelSymbolTable().getOp()->getContext())) {}

  /// Given a function, check to see if it is a top-level function.  If not,
  /// lower it to a ParamDeclareRegionOp.
  void lowerNestedFunction(FnOp func);
  /// Lower LIT dialect operations in a function body.
  void lowerLITOps(FnOp func, bool &hadErrors);

  /// Lower a function from LIT FnOp to KGEN GeneratorOp.
  /// Caller must handle removal of the original symbol and pass the
  /// pre-calculated mangled name as `newName` beforehand, since different
  /// contexts (top-level vs nested, struct methods) have different
  /// requirements. This function handles symbol table invalidation.
  LogicalResult lowerFunction(FnOp func,
                              ArrayRef<ParamDeclAttr> parentInputParams,
                              Block::iterator mainSymbolTablePosIter,
                              StringAttr newName);

  /// Lower lit.struct.decl and its nested structures.
  LogicalResult lowerStructDecl(StructDeclOp structDecl,
                                Block::iterator mainSymbolTablePosIter);
  /// Lower lit.extension.decl and its nested structures.
  LogicalResult lowerExtensionDecl(ExtensionDeclOp extensionDecl,
                                   Block::iterator mainSymbolTablePosIter);

  /// Lower lit.trait.decl and its nested structures.
  LogicalResult lowerTraitDecl(TraitDeclOp traitDecl,
                               Block::iterator mainSymbolTablePosIter);
  /// Lower the constructs within the body of a module decl.
  /// isTopLevel indicates whether operations in this module body are direct
  /// children of the top-level symbol table (true) or nested within other
  /// operations like FileModuleOp/PackageOp (false). This determines the
  /// removal strategy for operations.
  LogicalResult lowerModuleDecl(Block *moduleBody,
                                Block::iterator mainSymbolTablePosIter,
                                bool isTopLevel);

  /// Recursively process all structs in the module hierarchy.
  LogicalResult lowerAllStructs(Block *moduleBody,
                                Block::iterator mainSymbolTablePosIter,
                                bool isTopLevel);

  /// Recursively process all extensions in the module hierarchy.
  LogicalResult lowerAllExtensions(Block *moduleBody,
                                   Block::iterator mainSymbolTablePosIter,
                                   bool isTopLevel);

  SymbolTable &getTopLevelSymbolTable() {
    return symbolTables.getTopLevelSymbolTable();
  }

  mlir::SymbolTableCollection &getSymbolTableCollection() {
    return symbolTables.getSymbolTables();
  }

  mlir::SymbolTableAnalysis &symbolTables;
  DenseMap<StringAttr, StringAttr> &renamedSymbols;
  SingletonTypeHelper &singletonTypeHelper;
  StructDecls &structDecls;
  TypeType typeType;
  /// For each symbol name (post-rename), the param decls that were dropped.
  DenseMap<StringAttr, ParamDeclDropMask> symbolDroppedParamDecls;
  bool foundAnyPatterns = false;
};
} // namespace

void LITLowerer::lowerLITOps(FnOp func, bool &hadErrors) {
  func.getBodyRegion().walk([&](Operation *op) {
    // Lower any aliases within the function body to param declare.
    IRRewriter b{OpBuilder(op)};
    if (AliasDeclOp alias = dyn_cast<AliasDeclOp>(op)) {
      // Aliases are eagerly substituted for their value, so they are no longer
      // referenced anymore.
      op->erase();
    } else if (isa<OwnershipUseOp, OwnershipMarkInitializedOp,
                   OwnershipMarkDestroyedOp, OwnershipMarkConsumedOp, ImportOp,
                   UnresolvedImportOp, UnresolvedWildcardImportOp>(op)) {
      // lit.ownership.* are used internally by the
      // frontend and ownership lowering, but is not needed after that.
      op->erase();
    } else if (auto loadConsume = dyn_cast<LoadConsumeOp>(op)) {
      b.replaceOpWithNewOp<RefLoadOp>(loadConsume, loadConsume.getRef());
    } else if (auto call = dyn_cast<LIT::CallOp>(op)) {
      if (auto symbolCst = dyn_cast<SymbolConstantAttr>(call.getCallee())) {
        b.replaceOpWithNewOp<KGEN::CallOp>(call, call.getResultTypes(),
                                           symbolCst, call.getOperands(),
                                           call.getTailKindAttr());
      } else {
        b.replaceOpWithNewOp<KGEN::CallParamOp>(
            call, call.getResultTypes(), call.getCallee(), call.getOperands(),
            call.getTailKindAttr());
      }
    } else if (auto call = dyn_cast<LIT::CallIndirectOp>(op)) {
      b.replaceOpWithNewOp<KGEN::CallIndirectOp>(
          call, call.getResultTypes(), call.getCallee(), call.getArguments(),
          call.getTailKindAttr());
    } else if (auto bindParams = dyn_cast<BindParamsOp>(op)) {
      if (failed(lowerBindParamsOp(bindParams, singletonTypeHelper)))
        hadErrors = true;
    } else if (auto call = dyn_cast<LIT::AsyncCallOp>(op)) {
      b.replaceOpWithNewOp<CO::InvokeOp>(call, call.getCallee(),
                                         call.getOperands());
    } else if (auto returnOp = dyn_cast<ErrorReturnOp>(op)) {
      b.replaceOpWithNewOp<HLCF::ReturnOp>(returnOp, returnOp.getResult());
    } else if (auto funcOp = dyn_cast<FnOp>(op)) {
      lowerNestedFunction(funcOp);
    }
  });
}

/// Rename the given symbol operation to its flattened/mangled name, remove it
/// from its current position, and reinsert it at the specified location in the
/// symbol table. Returns the flattened symbol name.
template <typename T>
static StringAttr
flattenNameAndReinsertOp(T op, SymbolTable &symbolTable,
                         Block::iterator mainSymbolTablePosIter) {
  auto mangled = MangledSymbol::mangle(op);
  StringAttr name = mangled.mangled;
  // No mangling occurred.
  if (name == op.getNameAttr())
    return name;

  // Remove the operation in preparation for re-insertion. This gets handled
  // differently depending on if we are already tracking this op in the symbol
  // table.
  if (op->getParentOp() == symbolTable.getOp())
    symbolTable.remove(op);
  else
    op->remove();

  op.setSymbolName(mangled.mangled);
  symbolTable.insert(op, mainSymbolTablePosIter);
  return mangled.mangled;
}

LogicalResult
LITLowerer::lowerFunction(FnOp func, ArrayRef<ParamDeclAttr> parentInputParams,
                          Block::iterator mainSymbolTablePosIter,
                          StringAttr newName) {
  // Caller is responsible for removing `func` since removal patterns differ:
  // - Top-level functions: symbolTable.remove()
  //   (Top-level functions are tracked in the symbol table and must be removed
  //   via symbolTable.remove() to maintain symbol table integrity)
  // - Nested functions: op->remove()
  //   (Nested functions are not tracked in the symbol table and can be removed
  //   directly from their parent operation)
  // Invalidate the symbol table for this FnOp. All FnOps have the SymbolTable
  // trait, so the SymbolTableCollection needs to be notified before erasure.
  getSymbolTableCollection().invalidateSymbolTable(func);

  // If this function has a subprogram attached, update its information to
  // account for the new name.
  if (newName != func.getSymNameAttr()) {
    func.setSymbolName(newName);
    DebugInfo::updateSubprogram(func, newName);
  }

  bool hadErrors = false;
  lowerLITOps(func, hadErrors);
  if (hadErrors)
    return failure();

  FnTypeGeneratorType signature = func.getFuncTypeGenerator();

  // Build the parameter list of the new function, prepending the parameters
  // from the parent decl if present.
  SmallVector<ParamDeclAttr> inputParams;
  if (!parentInputParams.empty()) {
    // Concat the parent and generator input parameter decls.
    llvm::append_range(inputParams, parentInputParams);
    // Offset index references within the current signature to make room.
    // Remap parent input parameter references to indices.
    signature =
        FnTypeGeneratorType::prependParams(signature, parentInputParams);
  }
  llvm::append_range(inputParams, extractImplicitOriginParams(func));

  // Snapshot the source-declared parameter list (names + passing kinds +
  // variadic kinds) before `lowerAttributesAndTypes` strips metadata from
  // the live signature. This snapshot is the source of truth for reflection
  // queries; the live `funcTypeGenerator` and `inputParams` may be rewritten
  // by later transforms such as `RemoveUnusedParams`.
  //
  // Default values and constraints are dropped here because they can hold
  // `ParamIndexRefAttr`s that need a contextual signature; the op's attribute
  // dictionary doesn't establish one, so leaving them in would fail
  // `verify-parameters`. Reflection only needs the structural names today.
  PogListAttr sourceParamList;
  if (PogListAttr fullList = signature.getParamListAttrs()) {
    SmallVector<PogMetadataAttr> strippedPogs;
    strippedPogs.reserve(fullList.getPogs().size());
    for (PogMetadataAttr pog : fullList.getPogs()) {
      strippedPogs.push_back(PogMetadataAttr::get(
          pog.getName(), pog.getPassingKind(), pog.getVariadic()));
    }
    sourceParamList = PogListAttr::get(func->getContext(), strippedPogs,
                                       /*bodyConstraints=*/{},
                                       fullList.getOrigVariadicConvention());
  }

  // Now that we have the full parameter list, remove any singleton parameters.
  // This ensures that the elaborator doesn't instantiate the function based on
  // lifetimes.
  ParamDeclDropMask droppedParams =
      removeSingletonParamDecls(singletonTypeHelper, inputParams);
  if (droppedParams.any())
    symbolDroppedParamDecls[func.getSymNameAttr()] = droppedParams;

  OpBuilder b(func->getContext());
  auto inputParamsArr = ParamDeclArrayAttr::get(b.getContext(), inputParams);
  auto sigAttr = TypeAttr::get(signature);

  // Snapshot the full source signature into `sourceFuncTypeGenerator`, wrapped
  // in a `TypeParamAttr` so `lowerLITTypes` lowers it in the value domain (the
  // snapshot's argument/result types come out in the value domain, which
  // reflection clients need). `lowerAttributesAndTypes` strips its metadata
  // like the live signature's, but it is left untouched by later transforms
  // (e.g. `RemoveUnusedParams`) that rewrite the live one.
  TypedAttr sourceFuncTypeGen = TypeParamAttr::get(signature, typeType);

  // Directly lower since these operations are exactly identical right now.
  OperationState state(func.getLoc(), GeneratorOp::getOperationName());
  GeneratorOp::build(b, state, func.getSymNameAttr(),
                     /*sym_visibility=*/nullptr, func.getSourceNameAttr(),
                     sigAttr, func.getFunctionTypeAttr(), inputParamsArr,
                     func.getDecoratorsAttr(), func.getInlineLevelAttr(),
                     func.getExportKindAttr(), func.getExternalAttr(),
                     /*inlinedForm=*/nullptr, func.getLinkageNameAttr(),
                     func.getFnAttrs(), func.getFnArgAttrs(), sourceParamList,
                     sourceFuncTypeGen);

  for (const NamedAttribute &attr : func->getDialectAttrs())
    state.attributes.push_back(attr);

  auto newFunc = cast<GeneratorOp>(b.create(state));

  // Move over the body.
  newFunc.getBodyRegion().takeBody(func.getBodyRegion());

  // Insert the lowered GeneratorOp and cleanup the original FnOp.
  // Caller should have already removed the LIT function from its parent.
  getTopLevelSymbolTable().insert(newFunc, mainSymbolTablePosIter);
  func.erase();
  return success();
}

void LITLowerer::lowerNestedFunction(FnOp func) {
  // Process a nested function by lowering it straight to a
  // `kgen.param.declare.region`. Nested functions are denoted with an
  // parameter declaration on the function declaration.
  ParamDeclAttr decl = func.getParamDeclAttr();
  assert(decl && "expected nested function to declare a parameter");

  ImplicitLocOpBuilder b(func.getLoc(), func);

  // The new param.declare.region will drop implicit lifetimes.
  SmallVector<ParamDeclAttr> inputParams;
  llvm::append_range(inputParams, extractImplicitOriginParams(func));
  removeSingletonParamDecls(singletonTypeHelper, inputParams);

  StringAttr sourceName = func.getSourceNameAttr();
  if (!sourceName)
    sourceName = decl.getName();
  auto region = ParamDeclareRegionOp::create(
      b, /*sym_name=*/nullptr, /*sym_visibility=*/nullptr, decl, sourceName,
      func.getFuncTypeGenerator(), func.getFunctionType(), inputParams,
      func.getInlineLevelAttr(), func.getLinkageNameAttr(), func.getFnAttrs(),
      func.getFnArgAttrs());
  // The convenience builder only takes a level, so an unfolded
  // `@inline(expr)` has to be carried over separately.
  region.setInlineLevelAttr(func.getInlineLevelAttr());
  region.getBodyRegion().takeBody(func.getBodyRegion());
  func.erase();
}

LogicalResult
LITLowerer::lowerStructDecl(StructDeclOp structDecl,
                            Block::iterator mainSymbolTablePosIter) {
  // Update the name of this struct, incorporating any parents.
  StringAttr structName = flattenNameAndReinsertOp(
      structDecl, getTopLevelSymbolTable(), mainSymbolTablePosIter);

  // Build a StructGeneratorOp as its replacement.
  StructDecl info{};
  info.sourceName = structDecl.getSourceNameAttr();
  info.decls = structDecl.getParamsAttr();

  // Build the isMemoryOnly attribute. For unconditional RP, this is a simple
  // BoolAttr. For conditional RP, build a parametric expression that negates
  // the RP constraint proposition.
  auto *ctx = structDecl.getContext();
  if (structDecl.isRegisterPassable()) {
    info.isMemoryOnlyAttr = BoolAttr::get(ctx, false);
  } else if (auto rpConstraint =
                 structDecl.getRegisterPassableConstraintAttr()) {
    info.isMemoryOnlyAttr = ParamOperatorAttr::getNot(CastToBuiltinAttr::get(
        rpConstraint.getProposition(), IntegerType::get(ctx, 1)));
  } else {
    info.isMemoryOnlyAttr = BoolAttr::get(ctx, true);
  }

  // Provide default alignment of 1 if not explicitly specified.
  if (auto minAlign = structDecl.getMinAlignmentAttr())
    info.minAlignment = minAlign;
  else
    info.minAlignment =
        IntegerAttr::get(IndexType::get(structDecl.getContext()), 1);
  info.loc = structDecl.getLoc();

  // Collect the struct fields.
  SmallVector<StructDefFieldAttr> fieldDecls;
  for (auto [idx, field] : llvm::enumerate(structDecl.getFieldDecls())) {
    info.fields.emplace_back(field.getNameAttr(), field.getType());
    structDecls.fieldIndices.try_emplace({structName, field.getNameAttr()},
                                         idx);
    TypedAttr fieldTypeValue = TypeParamAttr::get(field.getType(), typeType);
    fieldDecls.push_back(StructDefFieldAttr::get(
        field.getNameAttr(), fieldTypeValue, field.getAnnotations()));
  }

  // Create struct-generator.
  SmallVector<StringAttr> paramNames;
  SmallVector<Type> paramTypes;
  SmallVector<TypedAttr> paramValues;
  for (ParamDeclAttr decl : info.decls) {
    paramNames.push_back(decl.getName());
    paramTypes.push_back(decl.getType());
    paramValues.push_back(ParamDeclRefAttr::get(decl));
  }

  auto structInstType = StructInstanceType::get(
      structName, paramNames, paramValues, fieldDecls, info.isMemoryOnlyAttr);

  OpBuilder b(structDecl->getContext());
  auto structGen = StructGeneratorOp::create(
      b, info.loc, structName, /*sym_visibility=*/nullptr, info.decls,
      structInstType, typeType, structDecl.getAnnotations());
  Block *structGenBody = b.createBlock(&structGen.getRegion());

  for (Operation &member : llvm::make_early_inc_range(
           structDecl.getFields().front().getOperations())) {
    if (isa<StructFieldOp>(member))
      continue; // Already lowered field.
    if (isa<AliasDeclOp>(member)) {
      member.erase();
      continue;
    }
    if (auto conformance = dyn_cast<ConformanceOp>(member)) {
      conformance->moveBefore(structGenBody, structGenBody->end());
      continue;
    }

    auto func = dyn_cast<FnOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    // Calculate new name, mangled if not top level. Must be before
    // removal since MangledSymbol::mangle crawls up the ancestors.
    StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
    // This is out here because removal is different for each
    // lowerFunction caller.
    func->remove();
    if (failed(lowerFunction(func, structDecl.getInputParams(),
                             mainSymbolTablePosIter, nameToUse)))
      return failure();
  }

  getTopLevelSymbolTable().remove(structDecl);
  info.symRef = SymbolRefAttr::get(
      getTopLevelSymbolTable().insert(structGen, mainSymbolTablePosIter));
  getSymbolTableCollection().invalidateSymbolTable(structDecl);
  structDecl.erase();
  structDecls.structDecls.try_emplace(structName, std::move(info));
  return success();
}

LogicalResult
LITLowerer::lowerExtensionDecl(ExtensionDeclOp extensionDecl,
                               Block::iterator mainSymbolTablePosIter) {
  SymbolRefAttr targetStructRef = extensionDecl.getTargetStruct().value();

  // Flatten the symbol reference to get the proper name for lookup
  StringAttr structName = flattenSymbolRefAttr(targetStructRef).getAttr();

  StructDecl &targetStructDeclInfo = structDecls.get(structName);

  Operation *kgenOp = getTopLevelSymbolTable().lookupSymbolIn(
      getTopLevelSymbolTable().getOp(), targetStructDeclInfo.symRef);
  if (!kgenOp) {
    return extensionDecl.emitError("cannot find extension target struct");
  }

  StructGeneratorOp kgenStructGenOp = dyn_cast<StructGeneratorOp>(kgenOp);
  if (!kgenStructGenOp) {
    return extensionDecl.emitError("extension target is not a struct");
  }

  for (Operation &member : llvm::make_early_inc_range(
           extensionDecl.getFields().front().getOperations())) {
    assert(!isa<StructFieldOp>(member) && "Extensions can't have fields");
    if (isa<AliasDeclOp>(member)) {
      member.erase();
      continue;
    }
    if (auto conformance = dyn_cast<ConformanceOp>(member)) {
      Block *structGenBody = &kgenStructGenOp.getRegion().front();
      // Extension conformances need to be moved to the target struct's
      // generator, because that's what the elaborator expects.
      conformance->moveBefore(structGenBody, structGenBody->end());
      continue;
    }

    auto func = dyn_cast<FnOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    ArrayRef<ParamDeclAttr> inputParams = targetStructDeclInfo.decls;
    StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
    func->remove();
    if (failed(lowerFunction(func, inputParams, mainSymbolTablePosIter,
                             nameToUse)))
      return failure();
  }
  // Invalidate symbol table before erasing to maintain consistency.
  getSymbolTableCollection().invalidateSymbolTable(extensionDecl);
  // Remove from symbol table if present, otherwise erase directly.
  // Note: Unlike StructDecl, we don't use flattenNameAndReinsertOp since we're
  // just erasing, not moving it first.
  // TODO(MOCO-522): Either move this first too, or change that about
  // structs/traits.
  if (getTopLevelSymbolTable().lookup(extensionDecl.getSymNameAttr())) {
    getTopLevelSymbolTable().erase(extensionDecl);
  } else {
    extensionDecl.erase();
  }
  return success();
}

LogicalResult
LITLowerer::lowerTraitDecl(TraitDeclOp traitDecl,
                           Block::iterator mainSymbolTablePosIter) {
  // Update the name of this trait, incorporating any parents.
  flattenNameAndReinsertOp(traitDecl, getTopLevelSymbolTable(),
                           mainSymbolTablePosIter);

  // Process operations within the trait body.
  for (Operation &member : llvm::make_early_inc_range(
           traitDecl.getFields().front().getOperations())) {
    if (auto func = dyn_cast<FnOp>(member)) {
      // Check if the function has a non-empty body (more than just
      // hlcf.unreachable).
      Block *funcBody = func.getBody();
      bool hasEmptyBody = funcBody->getOperations().size() == 1 &&
                          isa<HLCF::UnreachableOp>(funcBody->front());

      if (!hasEmptyBody) {
        // Calculate new name, mangled if not top level. Must be before
        // removal since MangledSymbol::mangle crawls up the ancestors.
        StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
        // This is out here because removal is different for each
        // lowerFunction caller.
        func->remove();
        if (failed(lowerFunction(func, traitDecl.getInputParams(),
                                 mainSymbolTablePosIter, nameToUse)))
          return failure();
      }
    }
    // We don't care about other operations in the trait body for now.
  }

  getTopLevelSymbolTable().erase(traitDecl);
  // invalidateSymbolTable since we're removing from the top-level
  // symbol table.
  getSymbolTableCollection().invalidateSymbolTable(traitDecl);
  return success();
}

LogicalResult
LITLowerer::lowerAllStructs(Block *moduleBody,
                            Block::iterator mainSymbolTablePosIter,
                            bool isTopLevel) {

  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    if (auto structDecl = dyn_cast<StructDeclOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          mainSymbolTablePosIter == Block::iterator() ? op.getIterator()
                                                      : mainSymbolTablePosIter;
      if (failed(lowerStructDecl(structDecl, childMainSymbolTablePos)))
        return failure();
    } else if (auto fileModule = dyn_cast<LIT::FileModuleOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          mainSymbolTablePosIter == Block::iterator() ? op.getIterator()
                                                      : mainSymbolTablePosIter;
      if (failed(lowerAllStructs(fileModule.getBody(), childMainSymbolTablePos,
                                 /*isTopLevel=*/false)))
        return failure();
    } else if (auto package = dyn_cast<LIT::PackageOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          mainSymbolTablePosIter == Block::iterator() ? op.getIterator()
                                                      : mainSymbolTablePosIter;
      if (failed(lowerAllStructs(package.getBody(), childMainSymbolTablePos,
                                 /*isTopLevel=*/false)))
        return failure();
    }
  }
  return success();
}

LogicalResult
LITLowerer::lowerAllExtensions(Block *moduleBody,
                               Block::iterator mainSymbolTablePosIter,
                               bool isTopLevel) {
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    if (auto extensionDecl = dyn_cast<ExtensionDeclOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      auto extensionPos =
          isTopLevel ? op.getIterator() : mainSymbolTablePosIter;
      if (failed(lowerExtensionDecl(extensionDecl, extensionPos)))
        return failure();
    } else if (auto fileModule = dyn_cast<LIT::FileModuleOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          isTopLevel ? op.getIterator() : mainSymbolTablePosIter;
      if (failed(lowerAllExtensions(fileModule.getBody(),
                                    childMainSymbolTablePos,
                                    /*isTopLevel=*/false)))
        return failure();
    } else if (auto package = dyn_cast<LIT::PackageOp>(op)) {
      // TODO(MOCO-522): Arcana docs on how we handle iterators in LowerLIT.
      Block::iterator childMainSymbolTablePos =
          isTopLevel ? op.getIterator() : mainSymbolTablePosIter;
      if (failed(lowerAllExtensions(package.getBody(), childMainSymbolTablePos,
                                    /*isTopLevel=*/false)))
        return failure();
    }
  }
  return success();
}

LogicalResult
LITLowerer::lowerModuleDecl(Block *moduleBody,
                            Block::iterator mainSymbolTablePosIter,
                            bool isTopLevel) {
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    // If we are already in the symbol table, use the the operations iterator.
    auto opSymTableIt = isTopLevel ? op.getIterator() : mainSymbolTablePosIter;

    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(&op)
            .Case([&](LIT::FnOp op) {
              // Calculate new name, mangled if not top level. Must be before
              // removal since MangledSymbol::mangle crawls up the ancestors.
              StringAttr nameToUse = !isTopLevel
                                         ? MangledSymbol::mangle(op).mangled
                                         : op.getSymNameAttr();
              // Function removal is handled at call site because top-level and
              // nested functions require different removal strategies.
              if (isTopLevel)
                getTopLevelSymbolTable().remove(op);
              else
                op->remove();
              return lowerFunction(op, {}, opSymTableIt, nameToUse);
            })
            .Case([&](StructDeclOp op) {
              // Structs should have been processed earlier by lowerAllStructs.
              assert(false && "Structs should have been lowered already");
              return failure();
            })
            .Case([&](ExtensionDeclOp op) {
              // Extensions should have been processed earlier by
              // lowerAllExtensions.
              assert(false && "Extensions should have been lowered already");
              return failure();
            })
            .Case([&](TraitDeclOp op) {
              return lowerTraitDecl(op, opSymTableIt);
            })
            .Case<LIT::FileModuleOp, LIT::PackageOp>([&](auto op) {
              // Make sure to remove the op from the symbol table if needed.
              if (op->getParentOp() == getTopLevelSymbolTable().getOp())
                getTopLevelSymbolTable().remove(op);

              // Lower the constructs within the body.
              Block *fileBody = op.getBody();
              if (failed(lowerModuleDecl(fileBody, opSymTableIt,
                                         /*isTopLevel=*/false)))
                return failure();

              // Inline the remaining body of the file into the parent.
              op->getBlock()->getOperations().splice(
                  op->getIterator(), fileBody->getOperations(),
                  fileBody->begin(), fileBody->end());

              // invalidateSymbolTable since we're removing from the top-level
              // symbol table.
              getSymbolTableCollection().invalidateSymbolTable(op);
              op->erase();
              return mlir::success();
            })
            .Case<AliasDeclOp, ImportOp, UnresolvedImportOp,
                  UnresolvedWildcardImportOp>([&](auto op) {
              op->erase();
              return mlir::success();
            })
            .Case([&](mlir::SymbolOpInterface symbol) {
              flattenNameAndReinsertOp(symbol, getTopLevelSymbolTable(),
                                       opSymTableIt);
              return mlir::success();
            })
            .Default(mlir::success());
    if (failed(result))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type lowering
//===----------------------------------------------------------------------===//

/// Check to see if any of the parameters of the specified signature are
/// singletons like origin parameters.  If so, bind them to a dummy value and
/// return the updated signature without them.
template <typename GenKind>
static std::conditional_t<std::is_base_of_v<Type, GenKind>, Type, Attribute>
removeSingletonParams(SingletonTypeHelper &singletonTypeHelper,
                      GenKind generator) {
  SmallVector<TypedAttr> paramsToBind;
  bool hasRemovals = false;

  for (Type paramType : generator.getInputParamTypes()) {
    if (singletonTypeHelper.isSingletonType(paramType)) {
      // Bind singleton parameters to their canonical value.
      paramsToBind.push_back(singletonTypeHelper.getSingletonValue(paramType));
      hasRemovals = true;
    } else {
      // Keep non-singleton parameters unbound.
      paramsToBind.push_back(UnboundAttr::get(paramType));
    }
  }

  // Update the generator if we dropped anything.
  if (hasRemovals) {
    generator = getSpecializedWithConcreteBindings(generator, paramsToBind);
    assert(generator && "didn't replace singletons correctly");
    if (generator.isFullyBound()) {
      // By back-compat, we never eliminate the empty generator type wrapper on
      // func types. This should eventually be made consistent with other types.
      // This follows the same pattern as BindParamsAttr.
      if constexpr (std::is_base_of_v<Type, GenKind>) {
        if (!isa<FuncType>(generator.getBody()))
          return generator.getBody();
      } else {
        if (!isa<FuncType>(generator.getBody().getType()))
          return generator.getBody();
      }
    }
  }
  return generator;
}

template <typename GenKind>
std::pair<std::conditional_t<std::is_base_of_v<Type, GenKind>, Type, Attribute>,
          WalkResult>
replaceGeneratorAttrType(SingletonTypeHelper &singletonTypeHelper,
                         mlir::AttrTypeReplacer &replacer, GenKind gen) {
  // Remove uses of any singleton attributes.
  SmallVector<Type> paramTypes;
  for (auto ty : gen.getInputParamTypes())
    paramTypes.push_back(replacer.replace(ty));

  // Remove metadata & remove singleton input param decls.
  auto newBody = cast<decltype(gen.getBody())>(replacer.replace(gen.getBody()));
  gen = GenKind::get(paramTypes, newBody);
  auto result = removeSingletonParams(singletonTypeHelper, gen);
  return std::make_pair(result, WalkResult::skip());
}

static LogicalResult lowerAttributesAndTypes(
    Operation *op, const DenseMap<StringAttr, StringAttr> &renamedSymbols,
    SingletonTypeHelper &singletonTypeHelper,
    DenseMap<StringAttr, ParamDeclDropMask> &symbolDroppedParamDecls,
    StructDecls &structDecls) {

  bool hadErrors = false;
  // This is the location of the current op that we're working on, updated as
  // we traverse the Module hierarchy.
  Location curOpLoc = op->getLoc();

  mlir::AttrTypeReplacer replacer;

  // OriginEqAttr may not survive to lowering.
  replacer.addReplacement(
      [&](OriginEqAttr attr)
          -> std::optional<std::pair<TypedAttr, WalkResult>> {
        mlir::emitError(curOpLoc)
            << "origin equality may only be tested in 'where' clauses";
        hadErrors = true;
        return std::make_pair(
            IntegerAttr::get(IntegerType::get(op->getContext(), 1), 0),
            WalkResult::skip());
      });

  // Member functions are reference with nested symbol references. After
  // lowering, the symbol tree will be flat. Concatenate all nested symbol
  // references in symbol constants. If something was renamed, perform the
  // renaming.
  replacer.addReplacement([&renamedSymbols](SymbolRefAttr ref) {
    auto flat = flattenSymbolRefAttr(ref);
    if (StringAttr renamed = renamedSymbols.lookup(flat.getAttr()))
      return SymbolRefAttr::get(renamed);
    return flat;
  });

  // Remove signature metadata.
  replacer.addReplacement([&](FuncType sig) {
    return std::make_pair(
        FuncType::get(cast<FunctionType>(replacer.replace(sig.getValues())),
                      sig.getArgConventions(), sig.getFnEffects()),
        WalkResult::skip());
  });

  replacer.addReplacement([&](GeneratorType gen) {
    // Remove uses of any singleton attributes.
    SmallVector<Type> paramTypes;
    for (auto ty : gen.getInputParamTypes())
      paramTypes.push_back(replacer.replace(ty));

    // Remove metadata & remove singleton input param decls.
    gen = GeneratorType::get(paramTypes, replacer.replace(gen.getBody()));
    auto result = removeSingletonParams(singletonTypeHelper, gen);
    return std::make_pair(result, WalkResult::skip());
  });

  replacer.addReplacement([&](GeneratorAttr gen) {
    return replaceGeneratorAttrType(singletonTypeHelper, replacer, gen);
  });

  replacer.addReplacement(
      [&](BindParamsAttr attr)
          -> std::optional<std::pair<TypedAttr, WalkResult>> {
        DenseBoolArrayAttr discharged = attr.getDischarged();
        if (!discharged || discharged.empty())
          return std::nullopt;

        // Remove discharge mask since it goes together with the generator
        // metadata's body constraints.
        TypedAttr generator =
            cast<TypedAttr>(replacer.replace(attr.getGenerator()));
        SmallVector<TypedAttr> paramValues;
        for (TypedAttr value : attr.getParamValues())
          paramValues.push_back(cast<TypedAttr>(replacer.replace(value)));
        return std::make_pair(
            BindParamsAttr::get(generator.getContext(), generator, paramValues,
                                /*evaluationContext=*/nullptr),
            WalkResult::skip());
      });

  replacer.addReplacement([&](StructInstanceType structInstType) {
    auto it = symbolDroppedParamDecls.find(structInstType.getName());
    if (it == symbolDroppedParamDecls.end() ||
        it->second.size() != structInstType.getParamValues().size())
      return std::make_pair(Type(structInstType), WalkResult::advance());

    SmallVector<StringAttr> remainingParamNames;
    SmallVector<TypedAttr> remainingParamValues;
    for (auto [idx, nameAndValue] :
         llvm::enumerate(llvm::zip(structInstType.getParamNames(),
                                   structInstType.getParamValues()))) {
      if (it->second[idx])
        continue;
      auto [name, value] = nameAndValue;
      remainingParamNames.push_back(name);
      remainingParamValues.push_back(cast<TypedAttr>(replacer.replace(value)));
    }

    SmallVector<StructDefFieldAttr> fields;
    fields.reserve(structInstType.getFields().size());
    for (StructDefFieldAttr field : structInstType.getFields()) {
      ArrayAttr replacedAnnotations;
      if (ArrayAttr annotations = field.getAnnotations()) {
        SmallVector<Attribute> replaced;
        for (AnnotationAttr annotation :
             annotations.getAsRange<AnnotationAttr>()) {
          replaced.push_back(AnnotationAttr::get(
              cast<TypedAttr>(replacer.replace(annotation.getValue())),
              cast<TypedAttr>(replacer.replace(annotation.getTypeValue()))));
        }
        replacedAnnotations =
            ArrayAttr::get(annotations.getContext(), replaced);
      }

      fields.push_back(StructDefFieldAttr::get(
          field.getName(),
          cast<TypedAttr>(replacer.replace(field.getTypeValue())),
          replacedAnnotations));
    }

    return std::make_pair(
        Type(StructInstanceType::get(structInstType.getName(),
                                     remainingParamNames, remainingParamValues,
                                     fields, structInstType.getIsMemoryOnly())),
        WalkResult::skip());
  });

  // Sugar attr is turned into canonical form.
  replacer.addReplacement(
      [&](SugarAttr sugar) { return replacer.replace(sugar.getCanonical()); });

  auto *debugInfoDialect =
      op->getContext()->getLoadedDialect<DebugInfo::DebugInfoDialect>();

  auto removeSingletonParams = [&](auto attr) -> decltype(attr) {
    SymbolRefAttr flatRef =
        cast<SymbolRefAttr>(replacer.replace(attr.getSymbol()));
    // Check the name & the number of params to ensure we don't operate on
    // SymbolConstantAttrs/FuncSymbolAttr that have already been processed.
    if (auto it = symbolDroppedParamDecls.find(flatRef.getLeafReference());
        it != symbolDroppedParamDecls.end() &&
        it->second.size() == attr.getParamValues().size()) {
      SmallVector<TypedAttr> remainingParams;
      for (auto [idx, value] : llvm::enumerate(attr.getParamValues()))
        if (!it->second[idx])
          remainingParams.push_back(cast<TypedAttr>(replacer.replace(value)));
      return decltype(attr)::get(
          flatRef,
          cast<decltype(attr.getType())>(replacer.replace(attr.getType())),
          remainingParams);
    }
    return nullptr;
  };

  replacer.addReplacement(
      [&](TypedAttr attr) -> std::optional<std::pair<TypedAttr, WalkResult>> {
        if (&attr.getDialect() == debugInfoDialect)
          return std::nullopt;

        // Canonicalize all values of singleton types.
        if (TypedAttr value =
                singletonTypeHelper.getSingletonValue(attr.getType()))
          return std::make_pair(value, WalkResult::advance());

        // Remove singleton parameter values from SymbolConstantAttr.
        if (auto symCst = dyn_cast<SymbolConstantAttr>(attr))
          if (auto newSymCst = removeSingletonParams(symCst))
            return std::make_pair(newSymCst, WalkResult::skip());

        // Remove singleton parameter values from FuncSymbolAttr.
        if (auto funcSym = dyn_cast<FuncSymbolAttr>(attr))
          if (auto newFuncSym = removeSingletonParams(funcSym))
            return std::make_pair(newFuncSym, WalkResult::skip());

        // Remove singleton parameter values from TypeGeneratorRefAttr.
        if (auto genRef = dyn_cast<TypeGeneratorRefAttr>(attr)) {
          SymbolRefAttr flatRef =
              cast<SymbolRefAttr>(replacer.replace(genRef.getSymbol()));
          if (auto it =
                  symbolDroppedParamDecls.find(flatRef.getLeafReference());
              it != symbolDroppedParamDecls.end() &&
              it->second.size() == genRef.getParamValues().size()) {
            SmallVector<TypedAttr> remainingParams;
            for (auto [idx, value] : llvm::enumerate(genRef.getParamValues()))
              if (!it->second[idx])
                remainingParams.push_back(
                    cast<TypedAttr>(replacer.replace(value)));
            return std::make_pair(
                TypeGeneratorRefAttr::get(attr.getContext(), flatRef,
                                          remainingParams, genRef.getType()),
                WalkResult::skip());
          }
        }

        // Remove singleton parameter values from BindParamsAttr.
        if (auto bindParams = dyn_cast<BindParamsAttr>(attr)) {
          SmallVector<TypedAttr> newOperands;
          for (auto [declType, param] : llvm::zip(
                   sugarCast<GeneratorType>(bindParams.getGenerator().getType())
                       .getInputParamTypes(),
                   bindParams.getParamValues())) {
            // Check for singleton type using the declared type on the
            // signature, instead of the concrete type of the param. This
            // prevents parametrically-singleton types from getting erased (only
            // always singleton params can be removed in general).
            if (!singletonTypeHelper.isSingletonType(
                    replacer.replace(declType)))
              newOperands.push_back(cast<TypedAttr>(replacer.replace(param)));
          }
          if (newOperands.size() != bindParams.getParamValues().size()) {
            TypedAttr generator =
                cast<TypedAttr>(replacer.replace(bindParams.getGenerator()));
            return std::make_pair(
                BindParamsAttr::get(generator.getContext(), generator,
                                    newOperands,
                                    /*evaluationContext=*/nullptr),
                WalkResult::skip());
          }
        }

        return std::nullopt;
      });

  // Walk the entire Module updating everything.
  op->walk([&](Operation *nestedOp) {
    curOpLoc = nestedOp->getLoc();
    replacer.replaceElementsIn(nestedOp, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
  });

  // Update saved types in struct decls.
  for (auto &decl : structDecls.structDecls) {
    decl.second.decls =
        cast<ParamDeclArrayAttr>(replacer.replace(decl.second.decls));
    for (auto &field : decl.second.fields) {
      field.second = replacer.replace(field.second);
    }
    decl.second.minAlignment =
        cast<TypedAttr>(replacer.replace(decl.second.minAlignment));
    decl.second.isMemoryOnlyAttr =
        cast<TypedAttr>(replacer.replace(decl.second.isMemoryOnlyAttr));
  }

  return failure(hadErrors);
}

// What follows lowers the remaining high-level `lit` types to KGEN. Notably,
// this eliminates symbol-based struct references in favor of `!kgen.struct`,
// turns `!lit.ref` into `!kgen.pointer`, and so on. It runs once the rest of
// the pass has lowered the module's declarations.

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
// Type lowering entrypoint
//===----------------------------------------------------------------------===//

static LogicalResult lowerLITTypes(ModuleOp module, StructDecls &state,
                                   mlir::LockedSymbolTableCollection &symtab) {
  // Reject the layouts that contain themselves by value.
  if (failed(analyzeStructRefs(state)))
    return failure();
  LITTypeLowerer b(module, state, symtab);

  // Lower operations first.
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<MaterializeIntoOp, StructInsertOp, LIT::StructExtractOp,
              RefImmutOp, RefUpcastOp, RefToPointerOp, RefFromPointerOp,
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

//===----------------------------------------------------------------------===//
// Pass boilerplate.
//===----------------------------------------------------------------------===//

namespace {
struct LowerLITPass : public KGEN::impl::LowerLITBase<LowerLITPass> {
  using LowerLITBase::LowerLITBase;

  void runOnOperation() override {
    // TODO: This has to be a module pass because this mutates the body of
    // the module, but we could trivially parallelize this within the pass.
    ModuleOp module = getOperation();
    auto &symtab = getAnalysis<mlir::SymbolTableAnalysis>();
    StructDecls structDecls;

    {
      DenseMap<StringAttr, StringAttr> renamedSymbols;
      SingletonTypeHelper singletonTypeHelper(
          module, symtab.getTopLevelSymbolTable(), structDecls);
      LITLowerer lowerer(symtab, renamedSymbols, singletonTypeHelper,
                         structDecls);

      // Lower all structs first, so that extensions can find them when they
      // need to look up struct info.
      if (failed(lowerer.lowerAllStructs(module.getBody(), Block::iterator(),
                                         /*isTopLevel=*/true)))
        return signalPassFailure();

      // Lower all extensions now that the structs' info exists.
      if (failed(lowerer.lowerAllExtensions(module.getBody(), Block::iterator(),
                                            /*isTopLevel=*/true)))
        return signalPassFailure();

      // Now lower away everything else including modules etc.
      if (failed(lowerer.lowerModuleDecl(module.getBody(), Block::iterator(),
                                         /*isTopLevel=*/true)) ||
          failed(lowerAttributesAndTypes(
              module, renamedSymbols, singletonTypeHelper,
              lowerer.symbolDroppedParamDecls, structDecls)))
        return signalPassFailure();
    }

    // Keep lowering all the operations and types.
    mlir::LockedSymbolTableCollection lockedSymtab(symtab.getSymbolTables());
    if (failed(lowerLITTypes(module, structDecls, lockedSymtab)))
      signalPassFailure();
  }
};

} // namespace
