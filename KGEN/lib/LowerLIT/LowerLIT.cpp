//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstraintSet.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLIT
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

static void buildDebugInfoValue(Operation *insertPt, Location loc,
                                StringRef varName,
                                DebugInfo::DIFileAttr fileAttr, Value value,
                                Type type) {
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
  auto varScope = DebugInfo::extractScope<DebugInfo::DILocalScopeAttr>(loc);
  if (!fileLoc || !varScope)
    return;

  auto varAttr = DebugInfo::DILocalVariableAttr::get(
      varScope, varName, fileAttr, fileLoc.getLine(), /*arg=*/0,
      /*alignInBits=*/0, DebugInfo::DIUnresolvedMLIRType::get(type));
  OpBuilder(insertPt).create<DebugInfo::ValueOp>(loc, value, varAttr);
}

/// Flatten the given symbol reference, collapsing all nested scopes into one
/// mangled name.
static FlatSymbolRefAttr flattenSymbolRefAttr(SymbolRefAttr ref) {
  // If the symbol is already flat, there is nothing to do.
  if (auto flatSym = dyn_cast<FlatSymbolRefAttr>(ref))
    return flatSym;

  // Flatten the symbol name into a single string.
  SmallString<32> name = ref.getRootReference().getValue();
  llvm::raw_svector_ostream nameOS(name);
  for (FlatSymbolRefAttr sym : ref.getNestedReferences())
    nameOS << "::" << sym.getValue();
  return SymbolRefAttr::get(ref.getContext(), nameOS.str());
}

static void lowerLITOps(LIT::FuncOp func,
                        DebugInfo::DISubprogramAttr funcSpAttr) {
  // Check if we are building debug info for source variables.
  bool buildingDebugVars =
      funcSpAttr && funcSpAttr.getCompileUnit().getEmissionKind() ==
                        DebugInfo::EmissionKind::Full;
  func.walk([&](Operation *op) {
    mlir::IRRewriter b{OpBuilder(op)};
    if (isa<AliasForwardDeclOp>(op)) {
      // lit.alias.fwd.decl is used internally by the frontend, but is not
      // needed by lowering at all.
      op->erase();
    } else if (auto letDecl = dyn_cast<LIT::LetRegDeclOp>(op)) {
      // Build information for this decl if necessary.
      if (buildingDebugVars) {
        buildDebugInfoValue(letDecl, letDecl.getLoc(), letDecl.getName(),
                            funcSpAttr.getFile(), letDecl.getOperand(),
                            letDecl.getType());
      }

      b.replaceOp(letDecl, letDecl.getOperand());
    } else if (auto varDecl = dyn_cast<LIT::VarLetDeclOp>(op)) {
      StringAttr varName = varDecl.getNameAttr();
      auto varType = varDecl.getType();

      // Lower a lit.varlet.decl to pop.stack_allocation.
      auto allocOp =
          b.replaceOpWithNewOp<POP::StackAllocationOp>(varDecl, varType, 1);

      // Build information for this variable if necessary.
      if (buildingDebugVars) {
        // TODO: Mark the value op as describing the "address" of the
        // variable, instead of claiming to describe the variable itself.
        buildDebugInfoValue(allocOp->getNextNode(), allocOp.getLoc(), varName,
                            funcSpAttr.getFile(), allocOp, varType);
      }
    }
  });
}

/// Flatten the name of the given symbol operation and insert it in the given
/// symbol table with that flattened name. Returns the flattened symbol name.
template <typename T>
static StringAttr flattenAndRenameSymbol(T op, const Twine &parentPrefix,
                                         SymbolTable &symbolTable,
                                         Block::iterator symbolTableIt) {
  StringAttr name = op.getSymNameAttr();
  if (parentPrefix.isTriviallyEmpty())
    return name;

  // Remove the operation in preparation for re-insertion. This gets handled
  // differently depending on if we are already tracking this op in the symbol
  // table.
  if (op->getParentOp() == symbolTable.getOp())
    symbolTable.remove(op);
  else
    op->remove();

  StringAttr newName =
      StringAttr::get(name.getContext(), parentPrefix + name.getValue());
  op.setSymNameAttr(newName);
  symbolTable.insert(op, symbolTableIt);
  return newName;
}

/// Lower an lit.func to kgen.generator.
static LogicalResult
lowerLITFunc(LIT::FuncOp gen, SymbolTable &symbolTable,
             Block::iterator symTableIt, const Twine &parentPrefix,
             ArrayRef<ParamDeclAttr> parentInputParams = {}) {
  auto funcSpAttr = DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(gen);

  // Update the function name, incorporating the parent prefix.
  if (!parentPrefix.isTriviallyEmpty()) {
    StringAttr newName =
        flattenAndRenameSymbol(gen, parentPrefix, symbolTable, symTableIt);

    // If this function has a subprogram attached, update its information to
    // account for the new name.
    if (funcSpAttr) {
      auto newSpAttr = DebugInfo::DISubprogramAttr::get(
          funcSpAttr.getContext(), funcSpAttr.getCompileUnit(),
          funcSpAttr.getScope(), funcSpAttr.getName(), newName,
          funcSpAttr.getFile(), funcSpAttr.getLine(), funcSpAttr.getScopeLine(),
          funcSpAttr.getSubprogramFlags(), funcSpAttr.getType());

      DebugInfo::DIAttrTypeReplacer replacer;
      replacer.addReplacement([&](DebugInfo::DISubprogramAttr attr) {
        return attr == funcSpAttr ? newSpAttr : attr;
      });
      replacer.recursivelyReplaceElementsIn(gen);
      funcSpAttr = newSpAttr;
    }
  }

  // Prepend the parameters from the parent decl if present.
  if (!parentInputParams.empty()) {
    SmallVector<ParamDeclAttr> paramDecls;
    ArrayRef<ParamDeclAttr> genParamDecls = gen.getInputParams();
    paramDecls.reserve(parentInputParams.size() + genParamDecls.size());
    llvm::append_range(paramDecls, parentInputParams);
    llvm::append_range(paramDecls, genParamDecls);

    gen.setSignature(SignatureType::get(
        ParamDeclArrayAttr::get(gen.getContext(), paramDecls),
        gen.getResultParamsAttr(), gen.getSignature().getValues(),
        gen.getMetadata()));
  }

  lowerLITOps(gen, funcSpAttr);
  OpBuilder b(gen);

  // Directly lower since these operations are exactly identical right now.
  auto result = b.create<GeneratorOp>(
      gen.getLoc(), gen.getSymNameAttr(), gen.getSignatureAttr(),
      gen.getConstraintsAttr(), gen.getAlwaysInlineLevelAttr());

  // Move over the body.
  auto *bodyBlock = gen.getBody();
  gen.getBodyRegion().getBlocks().remove(bodyBlock);
  result.getBodyRegion().push_back(bodyBlock);

  // Move over the symbol.
  symbolTable.erase(gen);
  gen = LIT::FuncOp(); // The line above also erases 'gen'.
  symbolTable.insert(result);

  return success();
}

/// Lower nested structures in lit.struct.decl away.
static LogicalResult lowerStructDecl(StructDeclOp structDecl,
                                     SymbolTable &symbolTable,
                                     Block::iterator symTableIt,
                                     const Twine &parentPrefix) {
  // Update the name of this struct, incorporating any parent prefix.
  StringAttr structName =
      flattenAndRenameSymbol(structDecl, parentPrefix, symbolTable, symTableIt);

  ArrayRef<ParamDeclAttr> structInputParams = structDecl.getInputParams();
  SmallVector<LIT::VarLetDeclOp> opsToErase;
  for (Operation &member : llvm::make_early_inc_range(
           structDecl.getFields().front().getOperations())) {
    if (isa<StructFieldOp>(member))
      continue; // Already lowered field.

    if (auto varDecl = dyn_cast<LIT::VarLetDeclOp>(member)) {
      Type elemType = ParamRefType::get(varDecl.getType().getElementType());
      OpBuilder b(&member);
      b.create<StructFieldOp>(member.getLoc(), varDecl.getName(), elemType);
      varDecl->erase();
      continue;
    } else if (auto paramDeclare = dyn_cast<KGEN::ParamDeclareOp>(member)) {
      paramDeclare.erase();
      continue;
    }
    auto func = dyn_cast<LIT::FuncOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    // Lower renamed function as usual.
    if (failed(lowerLITFunc(func, symbolTable, structDecl->getIterator(),
                            structName.getValue() + "::", structInputParams)))
      return failure();
  }
  return success();
}

static void lowerAttributesAndTypes(Operation *op) {
  mlir::AttrTypeReplacer replacer;

  // Member functions are reference with nested symbol references. After
  // lowering, the symbol tree will be flat. Concatenate all nested symbol
  // references in symbol constants.
  replacer.addReplacement(
      [](SymbolRefAttr ref) { return flattenSymbolRefAttr(ref); });

  // Lower `!lit.none` to `list<i1[0]>`, which will eventually become nothing.
  auto emptyList = ListType::get(IntegerType::get(op->getContext(), 1), 0);
  replacer.addReplacement([&](KGEN::LIT::NoneType type) { return emptyList; });
  // Lower `#lit.none` to `[]`.
  replacer.addReplacement([&](LIT::NoneAttr attr) {
    return ListAttr::get(attr.getContext(), {}, emptyList);
  });

  // Remove all input conventions and function effects.
  replacer.addReplacement([](MetadataAttr metadata) {
    return MetadataAttr::get(metadata.getContext(),
                             metadata.getInputConventions().size());
  });

  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);
}

/// Lower the constructs within the body of a module decl.
static LogicalResult lowerModuleDecl(Block *moduleBody,
                                     SymbolTable &symbolTable,
                                     Block::iterator symTableIt = {},
                                     const Twine &parentPrefix = {}) {
  bool isTopLevel = symTableIt == Block::iterator();
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    // If we are already in the symbol table, use the the operations iterator.
    auto opSymTableIt = isTopLevel ? op.getIterator() : symTableIt;

    if (auto func = dyn_cast<LIT::FuncOp>(op)) {
      if (failed(lowerLITFunc(func, symbolTable, opSymTableIt, parentPrefix)))
        return failure();
    } else if (auto structDecl = dyn_cast<StructDeclOp>(op)) {
      if (failed(lowerStructDecl(structDecl, symbolTable, opSymTableIt,
                                 parentPrefix)))
        return failure();
    } else if (auto fileDecl = dyn_cast<LIT::FileModuleOp>(op)) {
      // Lower the constructs within the body.
      Block *fileBody = fileDecl.getBody();
      if (failed(lowerModuleDecl(fileBody, symbolTable, opSymTableIt,
                                 parentPrefix + fileDecl.getName() + "::")))
        return failure();

      // Inline the remaining body of the file into the parent.
      fileDecl->getBlock()->getOperations().splice(
          fileDecl->getIterator(), fileBody->getOperations(), fileBody->begin(),
          fileBody->end());
      fileDecl->erase();
    } else if (isa<KGEN::ParamDeclareOp, LIT::UnresolvedImportOp>(op)) {
      op.erase();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass boilerplate.
//===----------------------------------------------------------------------===//

namespace {
struct LowerLITPass : public impl::LowerLITBase<LowerLITPass> {
  void runOnOperation() override {
    // TODO: This has to be a module pass because this mutates the body of
    // the module, but we could trivially parallelize this within the pass.
    ModuleOp module = getOperation();
    SymbolTable &symbolTable =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    if (failed(lowerModuleDecl(module.getBody(), symbolTable)))
      return signalPassFailure();
    lowerAttributesAndTypes(module);
  }
};

} // namespace
