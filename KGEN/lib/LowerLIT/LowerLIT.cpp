//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LowerLITTypes.h"

#include "KGEN/CODialect/COOps.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLIT
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Flatten the given symbol reference, collapsing all nested scopes into one
/// mangled name.
static FlatSymbolRefAttr flattenSymbolRefAttr(SymbolRefAttr ref) {
  // If the symbol is already flat, there is nothing to do.
  if (auto flatSym = dyn_cast<FlatSymbolRefAttr>(ref))
    return flatSym;

  // Flatten the symbol name into a single string.
  return FlatSymbolRefAttr::get(ref.getContext(), getFlattenedSymbolName(ref));
}

namespace {
/// Helper class for identifying singleton types.
/// Currently only considers a struct type as singleton if for all possible
/// parameter bindings, the struct type always yields a singleton type.
/// Supporting parametric singleton struct types is overkill right now.
class SingletonTypeHelper {
public:
  SingletonTypeHelper(ModuleOp &module, SymbolTable &symtab,
                      StructDecls &processedStructs)
      : module(module), symtab(symtab), processedStructs(processedStructs) {}

  bool isSingletonType(Type type);

  /// If `type` is a singleton type, this returns the singleton value of that
  /// type. Otherwise, returns a null result.
  TypedAttr getSingletonValue(Type type);

private:
  using StructCacheTy =
      DenseMap<StringAttr, SmallVector<std::tuple<StringAttr, TypedAttr>>>;

  /// Returns the iterator for the referenced struct decl into
  /// `alwaysSingletonStructs` (or end() if the decl is not always a singleton).
  StructCacheTy::iterator lookupStructSingletonFields(SymbolRefAttr ref);

  /// Determine the singleton-ness of a struct decl. This function always
  /// inserts the key into exactly one of:
  /// - alwaysSingletonStructs
  /// - notAlwaysSingletonStructs
  /// Returns the iterator for the key into `alwaysSingletonStructs` (or end()
  /// if it wasn't inserted there).
  /// This function is run at most once for each struct decl in the IR.
  StructCacheTy::iterator
  populateStructSingletonFields(StringAttr key,
                                ArrayRef<std::pair<StringAttr, Type>> fields);

  ModuleOp module;
  /// The SymbolTable contains all not-yet processed struct decls (unflattened),
  /// while StructDecls contains all struct decls already processed.
  SymbolTable &symtab;
  StructDecls &processedStructs;
  /// The keys are the struct decls that are always singletons (regardless
  /// of parameter bindings, if applicable). Each key is mapped to a list of
  /// singleton values that make up the singleton value of this struct. The
  /// exact singleton value in the form of a TypedAttr may differ depending
  /// on parameter bindings to the struct decl, so it cannot be produced
  /// until a concrete LIT::StructType is passed in.
  StructCacheTy alwaysSingletonStructs;
  /// These are the struct decls that are known to not always be singletons.
  DenseSet<StringAttr> notAlwaysSingletonStructs;

  /// Ephemeral data. Tracks the struct decls currently being checked in
  /// recursive `getSingletonValue` calls. If the same struct decl ever occurs
  /// again, it is not checked again, but immediately returns as _not_ a
  /// singleton type (since it is actually illegal, and will be caught later).
  DenseSet<StringAttr> inProgressStructs;
};
} // namespace

bool SingletonTypeHelper::isSingletonType(Type type) {
  if (isa<OriginType, OriginSetType>(type))
    return true;

  if (auto structType = dyn_cast<LIT::StructType>(type)) {
    return lookupStructSingletonFields(structType.getSymbol()) !=
           alwaysSingletonStructs.end();
  }
  return false;
}

TypedAttr SingletonTypeHelper::getSingletonValue(Type type) {
  if (auto origin = dyn_cast<OriginType>(type))
    return AnyOriginAttr::get(origin);
  if (auto set = dyn_cast<OriginSetType>(type))
    return OriginSetAttr::get(/*operands=*/{}, set);

  if (auto structType = dyn_cast<LIT::StructType>(type)) {
    auto it = lookupStructSingletonFields(structType.getSymbol());
    if (it == alwaysSingletonStructs.end())
      return {};
    return LITStructAttr::get(it->second, structType);
  }
  return {};
}

SingletonTypeHelper::StructCacheTy::iterator
SingletonTypeHelper::lookupStructSingletonFields(SymbolRefAttr ref) {
  StringAttr refName = flattenSymbolRefAttr(ref).getAttr();
  // If this struct is a known singleton, return its singleton value.
  if (auto it = alwaysSingletonStructs.find(refName);
      it != alwaysSingletonStructs.end())
    return it;

  // If this struct is a known non-singleton, return null attr.
  if (notAlwaysSingletonStructs.contains(refName))
    return alwaysSingletonStructs.end();

  // If we repeat an in-progress struct decl, it indicates an illegal cycle.
  // End the check and consider it _not_ a singleton type. LowerLITTypes will
  // report the error.
  if (!inProgressStructs.insert(refName).second)
    return {};

  // This is a struct decl we haven't seen before. Lookup its fields and
  // populate the cache with it.
  StructCacheTy::iterator outputIter;
  // First check the processed struct decls map.
  if (auto declIter = processedStructs.structDecls.find(refName);
      declIter != processedStructs.structDecls.end()) {
    outputIter =
        populateStructSingletonFields(refName, declIter->second.fields);
  } else {
    // If not already processed, the StructDeclOp must already exist in the
    // symbol table.
    Operation *op = symtab.lookupSymbolIn(module, ref);
    if (op == nullptr) {
      // This might be the struct currently being lowered, in which case, the
      // symbol name have been updated, yet the cache entry has not yet being
      // populated.
      op = symtab.lookupSymbolIn(module, refName);
    }
    StructDeclOp decl = cast<StructDeclOp>(op);
    SmallVector<std::pair<StringAttr, Type>> fields;
    for (StructFieldOp field : decl.getFieldDecls())
      fields.emplace_back(field.getNameAttr(), field.getType());
    outputIter = populateStructSingletonFields(refName, fields);
  }

  inProgressStructs.erase(refName);
  return outputIter;
}

SingletonTypeHelper::StructCacheTy::iterator
SingletonTypeHelper::populateStructSingletonFields(
    StringAttr key, ArrayRef<std::pair<StringAttr, Type>> fields) {
  SmallVector<std::tuple<StringAttr, TypedAttr>> values;
  for (auto [name, type] : fields) {
    TypedAttr value = getSingletonValue(type);
    if (!value) {
      notAlwaysSingletonStructs.insert(key);
      return alwaysSingletonStructs.end();
    }
    values.emplace_back(name, value);
  }
  return alwaysSingletonStructs.try_emplace(key, std::move(values)).first;
}

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
  void lowerLITOps(FnOp func);

  /// Lower a function from LIT FnOp to KGEN GeneratorOp.
  /// Caller must handle removal of the original symbol and pass the
  /// pre-calculated mangled name as `newName` beforehand, since different
  /// contexts (top-level vs nested, struct methods) have different
  /// requirements. This function handles symbol table invalidation.
  LogicalResult lowerFunction(FnOp func,
                              ArrayRef<ParamDeclAttr> parentInputParams,
                              ArrayRef<VariadicKind> parentVariadics,
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

void LITLowerer::lowerLITOps(FnOp func) {
  func.getBodyRegion().walk([&](Operation *op) {
    // Lower any aliases within the function body to param declare.
    IRRewriter b{OpBuilder(op)};
    if (AliasDeclOp alias = dyn_cast<AliasDeclOp>(op)) {
      // Aliases are eagerly substituted for their value, so they are no longer
      // referenced anymore.
      op->erase();
    } else if (auto lifetimeStart = dyn_cast<VarLifetimeStartOp>(op)) {
      auto arg = lifetimeStart.getArg();
      b.replaceOpWithNewOp<POP::StackAllocLifetimeStartOp>(
          op, arg.getDefiningOp()->getOperand(0));
    } else if (auto lifetimeEnd = dyn_cast<VarLifetimeEndOp>(op)) {
      auto arg = lifetimeEnd.getArg();
      b.replaceOpWithNewOp<POP::StackAllocLifetimeEndOp>(
          op, arg.getDefiningOp()->getOperand(0));
    } else if (isa<OwnershipUseOp, OwnershipMarkInitializedOp,
                   OwnershipMarkDestroyedOp, OwnershipMarkConsumedOp,
                   UnresolvedImportOp, UnresolvedWildcardImportOp>(op)) {
      // lit.ownership.* are used internally by the
      // frontend and ownership lowering, but is not needed after that.
      op->erase();
    } else if (auto loadConsume = dyn_cast<LoadConsumeOp>(op)) {
      b.replaceOpWithNewOp<RefLoadOp>(loadConsume, loadConsume.getRef());
    } else if (auto call = dyn_cast<LIT::CallOp>(op)) {
      if (auto symbolCst = dyn_cast<SymbolConstantAttr>(call.getCallee())) {
        b.replaceOpWithNewOp<KGEN::CallOp>(call, call.getResultTypes(),
                                           symbolCst, call.getOperands());
      } else {
        b.replaceOpWithNewOp<KGEN::CallParamOp>(
            call, call.getResultTypes(), call.getCallee(), call.getOperands());
      }
    } else if (auto call = dyn_cast<LIT::CallIndirectOp>(op)) {
      b.replaceOpWithNewOp<KGEN::CallIndirectOp>(
          call, call.getResultTypes(), call.getCallee(), call.getArguments());
    } else if (auto call = dyn_cast<LIT::AsyncCallOp>(op)) {
      b.replaceOpWithNewOp<CO::InvokeOp>(call, call.getCallee(),
                                         call.getOperands());
    } else if (auto varDecl = dyn_cast<VarDeclOp>(op)) {
      // Lower a lit.varlet.decl to pop.stack_allocation.
      auto allocOp = POP::StackAllocationOp::create(
          b, varDecl.getLoc(), /*markedLifetimes=*/true,
          varDecl.getType().getAsPointerType());

      // Replace !lit.ref result type with a cast from the pointer.  This will
      // get squashed by LowerLITTypes.
      b.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          varDecl, ArrayRef<Type>(varDecl.getType()), allocOp.getResult());
    } else if (auto returnOp = dyn_cast<ErrorReturnOp>(op)) {
      b.replaceOpWithNewOp<KGEN::ReturnOp>(returnOp, returnOp.getResult());
    } else if (auto elifOp = dyn_cast<HLCF::ElifOp>(op)) {
      HLCF::replaceElifWithIfOps(elifOp);
    } else if (auto funcOp = dyn_cast<FnOp>(op)) {
      lowerNestedFunction(funcOp);
    } else if (auto closureInit = dyn_cast<LIT::ClosureInitOp>(op)) {
      IRRewriter b{OpBuilder(closureInit)};
      SmallVector<Attribute> captureConventions =
          llvm::map_to_vector(closureInit.getCaptureConventions(),
                              [&](Attribute attr) -> Attribute {
                                if (isa<UnitAttr, MemSymbolTripleAttr>(attr))
                                  return attr;
                                return UnitAttr::get(attr.getContext());
                              });
      KGEN::ClosureInitOp closureInitKgen = KGEN::ClosureInitOp::create(
          b, closureInit.getLoc(), closureInit->getResults().front().getType(),
          closureInit.getFuncTypeGenerator(), closureInit.getFunctionType(),
          closureInit.getCaptures(),
          ArrayAttr::get(b.getContext(), captureConventions),
          closureInit.getInputParams().drop_front(
              closureInit.getFuncTypeGenerator().getNumImplicitOriginDecls()),
          closureInit.getInlineLevel(), closureInit.getNestedFnScopeAttr());
      closureInitKgen.getBodyRegion().takeBody(closureInit.getBodyRegion());
      b.replaceOp(closureInit, closureInitKgen);
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

  op.setName(mangled.mangled);
  symbolTable.insert(op, mainSymbolTablePosIter);
  return mangled.mangled;
}

LogicalResult
LITLowerer::lowerFunction(FnOp func, ArrayRef<ParamDeclAttr> parentInputParams,
                          ArrayRef<VariadicKind> parentVariadics,
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
    func.setName(newName);
    DebugInfo::updateSubprogram(func, newName);
  }

  lowerLITOps(func);

  FnTypeGeneratorType signature = func.getFuncTypeGenerator();

  // Build the parameter list of the new function, prepending the parameters
  // from the parent decl if present.
  SmallVector<ParamDeclAttr> inputParams;
  if (!parentInputParams.empty()) {
    // Concat the parent and generator input parameter decls.
    llvm::append_range(inputParams, parentInputParams);
    // Offset index references within the current signature to make room.
    // Remap parent input parameter references to indices.
    signature = FnTypeGeneratorType::prependParams(signature, parentInputParams,
                                                   parentVariadics);
  }
  llvm::append_range(inputParams, extractImplicitOriginParams(func));

  // If the function has an alias name, rename it.
  if (StringAttr newName = func.getLinkageNameAttr()) {
    renamedSymbols[func.getSymNameAttr()] = newName;
    func.setSymName(newName);
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

  // Directly lower since these operations are exactly identical right now.
  OperationState state(func.getLoc(), GeneratorOp::getOperationName());
  GeneratorOp::build(b, state, func.getSymNameAttr(), func.getSourceNameAttr(),
                     sigAttr, func.getFunctionTypeAttr(), inputParamsArr,
                     func.getDecoratorsAttr(), func.getInlineLevelAttr(),
                     func.getExportKindAttr(), func.getExternalAttr(),
                     /*inlinedForm=*/nullptr, func.getLLVMMetadataArray(),
                     func.getLLVMArgMetadataArray());

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
      b, decl, sourceName, func.getFuncTypeGenerator(), func.getFunctionType(),
      inputParams, func.getInlineLevel(), func.getLLVMMetadataArray(),
      func.getLLVMArgMetadataArray());
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
  info.isRegisterPassable = structDecl.isRegisterPassable();
  info.loc = structDecl.getLoc();

  // Collect the struct fields.
  SmallVector<StructDefFieldAttr> fieldDecls;
  for (auto [idx, field] : llvm::enumerate(structDecl.getFieldDecls())) {
    info.fields.emplace_back(field.getNameAttr(), field.getType());
    structDecls.fieldIndices.try_emplace({structName, field.getNameAttr()},
                                         idx);
    fieldDecls.push_back(
        StructDefFieldAttr::get(field.getNameAttr(), field.getType()));
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

  auto structInstType =
      StructInstanceType::get(structName, paramNames, paramValues, fieldDecls,
                              !info.isRegisterPassable);

  OpBuilder b(structDecl->getContext());
  auto structGen = StructGeneratorOp::create(
      b, info.loc, structName, info.decls, structInstType, typeType);
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
      // The trait decl is going away. This reference is no longer necessary.
      conformance.removeTraitRefAttr();
      conformance->moveBefore(structGenBody, structGenBody->end());
      continue;
    }

    auto func = dyn_cast<FnOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    // Lower renamed function as usual.
    SmallVector<VariadicKind> variadics = llvm::map_to_vector(
        structDecl.getSignature().getParamListAttrs().getPogs(),
        [](PogMetadataAttr pogAttr) { return pogAttr.getVariadic(); });

    // Calculate new name, mangled if not top level. Must be before
    // removal since MangledSymbol::mangle crawls up the ancestors.
    StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
    // This is out here because removal is different for each
    // lowerFunction caller.
    func->remove();
    if (failed(lowerFunction(func, structDecl.getInputParams(), variadics,
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
      // The trait decl is going away. This reference is no longer necessary.
      conformance.removeTraitRefAttr();
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
    SmallVector<VariadicKind> variadics(inputParams.size(), VariadicKind::None);

    StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
    func->remove();
    if (failed(lowerFunction(func, inputParams, variadics,
                             mainSymbolTablePosIter, nameToUse)))
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
      // kgen.unreachable).
      Block *funcBody = func.getBody();
      bool hasEmptyBody = funcBody->getOperations().size() == 1 &&
                          isa<KGEN::UnreachableOp>(funcBody->front());

      if (!hasEmptyBody) {
        // Lower the function with non-empty body.
        SmallVector<VariadicKind> variadics = llvm::map_to_vector(
            traitDecl.getSignature().getParamListAttrs().getPogs(),
            [](PogMetadataAttr pogAttr) { return pogAttr.getVariadic(); });

        // Calculate new name, mangled if not top level. Must be before
        // removal since MangledSymbol::mangle crawls up the ancestors.
        StringAttr nameToUse = MangledSymbol::mangle(func).mangled;
        // This is out here because removal is different for each
        // lowerFunction caller.
        func->remove();
        if (failed(lowerFunction(func, traitDecl.getInputParams(), variadics,
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

/// Add a link directive that shadows the package's name if the package was
/// precompiled. If this package is a source package, do nothing.
static LogicalResult addPackageLinkDirective(LIT::PackageOp package,
                                             SymbolTable &symtab) {
  // If the package wasn't compiled for anything, we currently treat it as a
  // "source package." This means that there are no link directives to insert.
  // FIXME: Once "source packages" no longer exist, insert a link directive
  // regardless, and compile for the build target on-demand.
  if (!package.getPostParseModuleAttr())
    return success();

  // We have at least some pre-compiled bytecode available, so insert a link
  // directive.
  OpBuilder b(package.getContext());
  auto linkOp = PackageLinkOp::create(
      b, package.getLoc(), package.getSymNameAttr(),
      package.getPostParseModuleAttr(), package.getDependenciesAttr());

  // Insert the link op into the symbol table right where the package was. Don't
  // erase the package op cause we need to do some cleanup still, but we do
  // still want to remove it from the symbol table.
  auto iter = package->getIterator();
  symtab.remove(package);
  symtab.insert(linkOp, iter);

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
              return lowerFunction(op, {}, {}, opSymTableIt, nameToUse);
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

              // If the package has already been compiled, insert a link
              // directive.
              if constexpr (std::is_same_v<decltype(op), LIT::PackageOp>)
                if (failed(
                        addPackageLinkDirective(op, getTopLevelSymbolTable())))
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
            .Case<AliasDeclOp, UnresolvedImportOp, UnresolvedWildcardImportOp>(
                [&](auto op) {
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
static GenKind removeSingletonParams(SingletonTypeHelper &singletonTypeHelper,
                                     GenKind signature) {
  llvm::SmallVector<TypedAttr> paramsToBind;
  size_t numRemoved = 0;

  ParameterEvaluator evaluator;
  for (auto [idx, paramType] :
       llvm::enumerate(signature.getInputParamTypes())) {
    Type adjParamType = evaluator.getReboundType(paramType);

    // If this is a parameter we are supposed to keep, leave it unbound.
    if (singletonTypeHelper.isSingletonType(paramType)) {
      // Bind the parameter to the expected singleton value of the right
      // type. 'getSpecializedSignature' strips off a level of indexes from
      // the type, so we need to adapt the type to cooperate.
      TypedAttr singletonValue =
          singletonTypeHelper.getSingletonValue(adjParamType);
      paramsToBind.push_back(singletonValue);
      evaluator.appendIndexBinding(paramsToBind.back());
      ++numRemoved;
    } else {
      // Any uses of this parameter in later replaced lifetimes needs to refer
      // to the appropriate index of the resultant parameter number, e.g. the
      // bool in a origin may shift to a new index.
      auto idxValue =
          ParamIndexRefAttr::get(/*depth*/ -1, idx - numRemoved, adjParamType);
      evaluator.appendIndexBinding(idxValue);

      // We tell getSpecializedSignature not to touch this though.
      paramsToBind.push_back(UnboundAttr::get(adjParamType));
    }
  }

  // Update the signature type if we dropped anything.
  if (numRemoved) {
    signature = signature.getSpecializedGenerator(
        paramsToBind, /*evaluationContext=*/nullptr);
    assert(signature && "didn't replace lifetimes correctly");
  }
  return signature;
}

template <typename GenKind>
std::pair<GenKind, WalkResult>
replaceGeneratorAttrType(SingletonTypeHelper &singletonTypeHelper,
                         mlir::AttrTypeReplacer &replacer, GenKind gen) {
  // Remove uses of any singleton attributes.
  SmallVector<Type> paramTypes;
  for (auto ty : gen.getInputParamTypes())
    paramTypes.push_back(replacer.replace(ty));

  // Remove metadata & remove singleton input param decls.
  auto newBody = cast<decltype(gen.getBody())>(replacer.replace(gen.getBody()));
  gen = GenKind::get(paramTypes, newBody);
  gen = removeSingletonParams(singletonTypeHelper, gen);
  return std::make_pair(gen, WalkResult::skip());
}

static void lowerAttributesAndTypes(
    Operation *op, const DenseMap<StringAttr, StringAttr> &renamedSymbols,
    SingletonTypeHelper &singletonTypeHelper,
    DenseMap<StringAttr, ParamDeclDropMask> &symbolDroppedParamDecls,
    StructDecls &structDecls) {
  mlir::AttrTypeReplacer replacer;

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
    gen = removeSingletonParams(singletonTypeHelper, gen);
    return std::make_pair(gen, WalkResult::skip());
  });

  replacer.addReplacement([&](GeneratorAttr gen) {
    return replaceGeneratorAttrType(singletonTypeHelper, replacer, gen);
  });

  auto *debugInfoDialect =
      op->getContext()->getLoadedDialect<DebugInfo::DebugInfoDialect>();
  replacer.addReplacement(
      [&](TypedAttr attr) -> std::optional<std::pair<TypedAttr, WalkResult>> {
        if (&attr.getDialect() == debugInfoDialect)
          return std::nullopt;

        // Canonicalize all values of singleton types.
        if (TypedAttr value =
                singletonTypeHelper.getSingletonValue(attr.getType()))
          return std::make_pair(value, WalkResult::advance());

        // Remove singleton parameter values from SymbolConstantAttr.
        if (auto symCst = dyn_cast<SymbolConstantAttr>(attr)) {
          SymbolRefAttr flatRef =
              cast<SymbolRefAttr>(replacer.replace(symCst.getSymbol()));
          // Check the name & the number of params to ensure we don't operate on
          // SymbolConstantAttrs that have already been processed.
          if (auto it =
                  symbolDroppedParamDecls.find(flatRef.getLeafReference());
              it != symbolDroppedParamDecls.end() &&
              it->second.size() == symCst.getParamValues().size()) {
            SmallVector<TypedAttr> remainingParams;
            for (auto [idx, value] : llvm::enumerate(symCst.getParamValues()))
              if (!it->second[idx])
                remainingParams.push_back(
                    cast<TypedAttr>(replacer.replace(value)));
            return std::make_pair(
                SymbolConstantAttr::get(flatRef,
                                        cast<FuncTypeGeneratorType>(
                                            replacer.replace(symCst.getType())),
                                        remainingParams),
                WalkResult::skip());
          }
        }

        // Remove singleton parameter values from BindParamsAttr.
        if (auto bindParams = dyn_cast<BindParamsAttr>(attr)) {
          SmallVector<TypedAttr> newOperands;
          for (auto [declType, param] : llvm::zip(
                   cast<GeneratorType>(bindParams.getGenerator().getType())
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
          if (newOperands.size() != bindParams.getParamValues().size())
            return std::make_pair(
                BindParamsAttr::get(cast<TypedAttr>(replacer.replace(
                                        bindParams.getGenerator())),
                                    newOperands,
                                    /*evaluationContext=*/nullptr),
                WalkResult::skip());
        }

        return std::nullopt;
      });

  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);

  // Update saved types in struct decls.
  for (auto &decl : structDecls.structDecls) {
    decl.second.decls =
        cast<ParamDeclArrayAttr>(replacer.replace(decl.second.decls));
    for (auto &field : decl.second.fields) {
      field.second = replacer.replace(field.second);
    }
  }
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
                                         /*isTopLevel=*/true)))
        return signalPassFailure();
      lowerAttributesAndTypes(module, renamedSymbols, singletonTypeHelper,
                              lowerer.symbolDroppedParamDecls, structDecls);
    }

    // Keep lowering all the operations and types.
    mlir::LockedSymbolTableCollection lockedSymtab(symtab.getSymbolTables());
    if (failed(LIT::lowerLITTypes(module, structDecls, lockedSymtab)))
      signalPassFailure();
  }
};

} // namespace
