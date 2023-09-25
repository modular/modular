//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstraintSet.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <deque>

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLIT
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

static void buildDebugInfoValue(OpBuilder &b, Operation *op, StringRef varName,
                                DebugInfo::DIFileAttr fileAttr, Value value,
                                Type type) {
  Location loc = op->getLoc();
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
  if (!fileLoc)
    return;
  ErrorOr<DebugInfo::DIScopeAttr> scopeOr = DebugInfo::getScopeWithinBody(loc);
  if (scopeOr.isError())
    return;
  auto varScope = dyn_cast_or_null<DebugInfo::DILocalScopeAttr>(*scopeOr);
  if (!varScope)
    return;

  auto varAttr = DebugInfo::DILocalVariableAttr::get(
      varScope, varName, fileAttr, fileLoc.getLine(), /*arg=*/0,
      /*alignInBits=*/0, DebugInfo::DIUnresolvedMLIRType::get(type));
  b.create<DebugInfo::ValueOp>(loc, value, varAttr);
}

/// Flatten the given symbol reference, collapsing all nested scopes into one
/// mangled name.
static FlatSymbolRefAttr flattenSymbolRefAttr(SymbolRefAttr ref) {
  // If the symbol is already flat, there is nothing to do.
  if (auto flatSym = dyn_cast<FlatSymbolRefAttr>(ref))
    return flatSym;

  // Flatten the symbol name into a single string.
  return SymbolRefAttr::get(ref.getContext(), getFlattenedSymbolName(ref));
}

static void lowerHandleVariant(HandleVariantOp handleVariantOp) {
  TypedValue<POP::VariantType> variantOperand = handleVariantOp.getVariant();
  mlir::IRRewriter b{OpBuilder(handleVariantOp)};
  Type successType = variantOperand.getType().getType(1);
  auto variantIsOp = b.create<POP::VariantIsOp>(handleVariantOp.getLoc(),
                                                variantOperand, successType);
  auto ifOp = b.create<HLCF::IfOp>(handleVariantOp.getLoc(),
                                   TypeRange(successType), variantIsOp);
  ifOp.getThenRegion().takeBody(handleVariantOp.getSuccessRegion());
  ifOp.getElseRegion().takeBody(handleVariantOp.getErrorRegion());
  if (auto litYield = dyn_cast<LIT::YieldOp>(
          ifOp.getThenRegion().front().getTerminator())) {
    mlir::IRRewriter b{OpBuilder(litYield)};
    b.replaceOpWithNewOp<HLCF::YieldOp>(litYield, litYield->getOperands());
  }
  b.replaceAllUsesWith(handleVariantOp.getResult(0), ifOp->getResult(0));
  handleVariantOp->erase();
}

/// Given a function, check to see if it is a top-level function.  If not, lower
/// it to a ParamDeclareRegionOp.
static void lowerNestedFunction(LIT::FuncOp func) {
  // Process a nested function by lowering it straight to a
  // `kgen.param.declare.region`. Nested functions are denoted with an
  // parameter declaration on the function declaration.
  ParamDeclAttr decl = func.getParamDeclAttr();
  assert(decl && "expected nested function to declare a parameter");

  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(func));
  auto region = b.create<ParamDeclareRegionOp>(
      decl, func.getSignature(), func.getFunctionType(), func.getInputParams(),
      func.getResultParams(), ArrayRef<ConstraintAttr>(),
      /*isolated=*/false, func.getInlineLevel());
  region.getBodyRegion().takeBody(func.getBodyRegion());
  func.erase();
}

static void lowerLITOps(LIT::FuncOp func) {
  // Check if we are building debug info for source variables.
  DebugInfo::DISubprogramAttr funcSpAttr = func.getSubprogramScope();
  bool buildingDebugVars =
      funcSpAttr && funcSpAttr.getCompileUnit().getEmissionKind() ==
                        DebugInfo::EmissionKind::Full;
  func.getBodyRegion().walk([&](Operation *op) {
    // Lower any aliases within the function body to param declare.
    if (AliasDeclOp alias = dyn_cast<AliasDeclOp>(op)) {
      mlir::IRRewriter b{OpBuilder(op)};
      b.replaceOpWithNewOp<ParamDeclareOp>(
          alias, TypeRange(), alias.getParamDecl(), alias.getValue());
    } else if (isa<AliasForwardDeclOp, OwnershipUseOp, OwnershipMarkDestroyedOp,
                   OwnershipDefLValueOp>(op)) {
      // lit.alias.fwd_decl and lit.ownership.* are used internally by the
      // frontend and ownership lowering, but is not needed after that.
      op->erase();
    } else if (isa<OwnershipEndLifetimeOp, OwnershipMakePointerLValue>(op)) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    } else if (auto loadConsume = dyn_cast<LoadConsumeOp>(op)) {
      mlir::IRRewriter b{OpBuilder(op)};
      b.replaceOpWithNewOp<POP::LoadOp>(loadConsume, loadConsume.getPtr());
    } else if (auto letDecl = dyn_cast<LetRegDeclOp>(op)) {
      mlir::IRRewriter b{OpBuilder(op)};
      // Build information for this decl if necessary.
      if (buildingDebugVars) {
        buildDebugInfoValue(b, letDecl, letDecl.getName(), funcSpAttr.getFile(),
                            letDecl.getOperand(), letDecl.getType());
      }

      b.replaceOp(letDecl, letDecl.getOperand());
    } else if (auto varDecl = dyn_cast<VarLetDeclOp>(op)) {
      mlir::IRRewriter b{OpBuilder(op)};
      StringAttr varName = varDecl.getNameAttr();
      auto varType = varDecl.getType().getAsPointerType();

      // Declare the lifetime used in the result type.
      b.create<ParamDeclareOp>(varDecl.getLoc(), varDecl.getParamDecl(),
                               b.getAttr<LifetimeAttr>());
      // Lower a lit.varlet.decl to pop.stack_allocation.
      auto allocOp =
          b.create<POP::StackAllocationOp>(varDecl.getLoc(), varType, 1);
      // Replace !lit.ref result type with a cast from the pointer.  This will
      // get squashed by LowerStructs.
      b.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          varDecl, ArrayRef<Type>(varDecl.getType()), allocOp.getResult());

      // Build information for this variable if necessary.
      if (buildingDebugVars) {
        // TODO: Mark the value op as describing the "address" of the
        // variable, instead of claiming to describe the variable itself.
        b.setInsertionPointAfter(allocOp);
        buildDebugInfoValue(b, allocOp, varName, funcSpAttr.getFile(), allocOp,
                            varType);
      }
    } else if (auto handleVariant = dyn_cast<HandleVariantOp>(op)) {
      lowerHandleVariant(handleVariant);
    } else if (auto returnOp = dyn_cast<ErrorReturnOp>(op)) {
      mlir::IRRewriter b{OpBuilder(op)};
      b.replaceOpWithNewOp<KGEN::ReturnOp>(returnOp, returnOp.getVariant());
    } else if (auto globalRefOp = dyn_cast<GlobalVarRefOp>(op)) {
      mlir::IRRewriter b{OpBuilder(op)};
      b.replaceOpWithNewOp<GlobalAddressOp>(globalRefOp, globalRefOp.getType(),
                                            globalRefOp.getGlobal());
    } else if (auto funcOp = dyn_cast<LIT::FuncOp>(op)) {
      lowerNestedFunction(funcOp);
    }
  });
}

/// Flatten the name of the given symbol operation and insert it in the given
/// symbol table with that flattened name. Returns the flattened symbol name.
template <typename T>
static StringAttr flattenAndRenameSymbol(T op, SymbolTable &symbolTable,
                                         Block::iterator symbolTableIt) {
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
  symbolTable.insert(op, symbolTableIt);
  return mangled.mangled;
}

/// Lower an lit.func to kgen.generator.
static LogicalResult
lowerLITFunc(LIT::FuncOp gen, SymbolTable &symbolTable,
             DenseMap<StringAttr, StringAttr> &renamedSymbols,
             Block::iterator symTableIt, const Twine &parentPrefix,
             ArrayRef<ParamDeclAttr> parentInputParams = {}) {
  // Update the function name, incorporating the parent prefix.
  if (!parentPrefix.isTriviallyEmpty()) {
    StringAttr newName = flattenAndRenameSymbol(gen, symbolTable, symTableIt);

    // If this function has a subprogram attached, update its information to
    // account for the new name.
    DebugInfo::updateSubprogram(gen, newName);
  }

  lowerLITOps(gen);

  ParamDeclArrayAttr inputParams = gen.getInputParamsAttr();
  SignatureType signature = gen.getSignature();
  // Prepend the parameters from the parent decl if present.
  if (!parentInputParams.empty()) {
    // Concat the parent and generator input parameter decls.
    SmallVector<ParamDeclAttr> paramDecls;
    paramDecls.reserve(parentInputParams.size() + inputParams.size());
    llvm::append_range(paramDecls, parentInputParams);
    llvm::append_range(paramDecls, inputParams);
    inputParams = ParamDeclArrayAttr::get(gen.getContext(), paramDecls);

    // Offset index references within the current signature to make room.
    // Remap parent input parameter references to indices.
    signature = IndexRefRemapper::prependParams(signature, parentInputParams);
  }

  // If the function has an alias name, rename it.
  if (StringAttr newName = gen.getLinkageNameAttr()) {
    renamedSymbols[gen.getNameAttr()] = newName;
    gen.setSymName(newName);
  }

  OpBuilder b(gen->getContext());
  Operation *result;
  if (gen.isExternal()) {
    // Replace external functions with `kgen.extern.generator` ops.
    result = b.create<ExternGeneratorOp>(
        gen.getLoc(), gen.getSymNameAttr(), TypeAttr::get(signature),
        gen.getFunctionTypeAttr(), inputParams, gen.getResultParamsAttr(),
        gen.getExportKindAttr(), gen.getPreCompiledModuleRefAttr());
  } else {
    // Directly lower since these operations are exactly identical right now.
    auto newGen = b.create<GeneratorOp>(
        gen.getLoc(), gen.getSymNameAttr(), TypeAttr::get(signature),
        gen.getFunctionTypeAttr(), inputParams, gen.getResultParamsAttr(),
        gen.getConstraintsAttr(), gen.getDecoratorsAttr(),
        gen.getInlineLevelAttr(), gen.getExportKindAttr());
    result = newGen;

    // Move over the body.
    auto *bodyBlock = gen.getBody();
    gen.getBodyRegion().getBlocks().remove(bodyBlock);
    newGen.getBodyRegion().push_back(bodyBlock);
  }

  // Move over the symbol, and we're done.
  Block::iterator genIter = gen->getIterator();
  symbolTable.remove(gen);
  symbolTable.insert(result, genIter);
  gen.erase();

  return success();
}

/// Lower nested structures in lit.struct.decl away.
static LogicalResult
lowerStructDecl(StructDeclOp structDecl, SymbolTable &symbolTable,
                DenseMap<StringAttr, StringAttr> &renamedSymbols,
                Block::iterator symTableIt) {
  // Update the name of this struct, incorporating any parents.
  StringAttr structName =
      flattenAndRenameSymbol(structDecl, symbolTable, symTableIt);

  for (Operation &member : llvm::make_early_inc_range(
           structDecl.getFields().front().getOperations())) {
    if (isa<StructFieldOp>(member))
      continue; // Already lowered field.
    if (isa<AliasDeclOp>(member)) {
      member.erase();
      continue;
    }

    auto func = dyn_cast<LIT::FuncOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    // Lower renamed function as usual.
    if (failed(lowerLITFunc(
            func, symbolTable, renamedSymbols, structDecl->getIterator(),
            structName.getValue() + "::", structDecl.getInputParams())))
      return failure();
  }
  return success();
}

static void lowerAttributesAndTypes(
    Operation *op, const DenseMap<StringAttr, StringAttr> &renamedSymbols) {
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

  // Lower `!lit.none` to `list<i1[0]>`, which will eventually become nothing.
  auto emptyList =
      POP::ArrayType::get(0, IntegerType::get(op->getContext(), 1));
  replacer.addReplacement([&](LIT::NoneType type) { return emptyList; });
  // Lower `#lit.none` to `[]`.
  replacer.addReplacement([&](LIT::NoneAttr attr) {
    return POP::ArrayAttr::get(attr.getContext(), {}, emptyList);
  });

  // Remove signature metadata.
  replacer.addReplacement([](SignatureType sig) {
    return SignatureType::get(sig.getValues(), sig.getInputParamTypes(),
                              sig.getResultParamTypes(), {},
                              sig.getFnEffects());
  });

  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);
}

/// Global variables have initializers that can reference other global
/// variables. This function will pass over all global variable declarations and
/// determine an initialization order based on the reference graph between
/// global variables in their initializers. It is not possible for the parser to
/// generate global variables that reference each other in a cycle.
static LogicalResult
orderAndLowerGlobalVariables(ModuleOp module,
                             DenseMap<StringAttr, StringAttr> &renamedSymbols) {
  struct GlobalRefNode {
    unsigned numRefs = 0;
    unsigned numReady = 0;
    SmallVector<GlobalRefNode *> refdBy;
    GlobalVarDeclOp op = nullptr;
  };

  DenseMap<SymbolRefAttr, std::unique_ptr<GlobalRefNode>> state;
  auto getOrCreate = [&](SymbolRefAttr ref) -> GlobalRefNode & {
    std::unique_ptr<GlobalRefNode> &cur = state[ref];
    if (!cur)
      cur = std::make_unique<GlobalRefNode>();
    return *cur;
  };

  module.walk([&](Operation *op) {
    if (auto global = dyn_cast<GlobalVarDeclOp>(op)) {
      // Ensure a node has been created for this global.
      SymbolRefAttr symbol = getFullyResolvedSymbolRef(global);
      GlobalRefNode &cur = getOrCreate(symbol);
      cur.op = global;
      // Find references to other globals.
      global.walk([&](GlobalVarRefOp ref) {
        // Global variables are allowed to reference themselves.
        if (ref.getGlobal() == symbol)
          return;
        GlobalRefNode &refNode = getOrCreate(ref.getGlobal());
        ++cur.numRefs;
        refNode.refdBy.push_back(&cur);
      });
      // No global variable declarations inside the bodies.
      return WalkResult::skip();
    }

    // Skip over the bodies of operations where global variables cannot exist.
    if (isa<LIT::StructDeclOp, LIT::FuncOp>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });

  // Process nodes breadth-first.
  std::deque<GlobalRefNode *> queue;
  uint32_t initOrder = 0;
  for (auto &[_, node] : state)
    if (node->numRefs == node->numReady)
      queue.push_back(node.get());

  while (!queue.empty()) {
    GlobalRefNode *node = queue.front();
    queue.pop_front();

    GlobalVarDeclOp op = node->op;
    mlir::IRRewriter b{OpBuilder(op)};
    MLIRContext *ctx = op.getContext();

    // Prepare locations and names for outlining the constructor and destructor.
    StringRef name = op.getSymName();
    auto ctorName = b.getStringAttr("(ctor_fn)" + name);
    auto dtorName = b.getStringAttr("(dtor_fn)" + name);
    Location ctorLoc = op->getLoc();
    Location dtorLoc = op->getLoc();
    if (DebugInfo::DIScopeAttr scope = op.getLocScope()) {
      // GlobalVarDeclOp either has a file scope or no scope.
      auto fileAttr = cast<DebugInfo::DIFileAttr>(scope);

      // TODO: "C" as the Dwarf language type is the best option for now.
      DebugInfo::DIBuilder dib(ctx);
      dib.initializeCompileUnit(llvm::dwarf::DW_LANG_C, fileAttr, "kgen",
                                /*isOptimized=*/true,
                                DebugInfo::EmissionKind::Full);

      // We set the scoped location for the outlined methods.
      auto spType = DebugInfo::DISubroutineType::get(ctx, {}, {});
      auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
      DebugInfo::DIBuilder::ScopeGuard guard = dib.pushScopeGuard(fileAttr);
      auto getXtorLoc = [&](StringAttr xtorName) {
        guard = dib.pushSubprogram(
            xtorName, xtorName, fileAttr, fileLoc.getLine(), fileLoc.getLine(),
            DebugInfo::SubprogramFlags::Definition, spType);
        return dib.createScopedLoc(fileLoc);
      };
      ctorLoc = getXtorLoc(ctorName);
      dtorLoc = getXtorLoc(dtorName);
    }

    // Outline the constructor and destructor into functions.
    auto sig = LITSignatureType::get(b.getContext());
    auto makeXtor = [&](Location xtorLoc, StringAttr xtorName, Region &body) {
      b.setInsertionPoint(op);
      auto fn = b.create<LIT::FuncOp>(xtorLoc, xtorName, sig);
      fn.getBodyRegion().takeBody(body);

      // If we have a debuginfo scope available, we update the ops in the body.
      if (auto sp = fn.getSubprogramScope()) {
        fn.getBodyRegion().walk([&](Operation *op) {
          op->setLoc(mlir::FusedLoc::get(
              ctx, {op->getLoc()->findInstanceOf<FileLineColLoc>()}, sp));
        });
      }
      b.setInsertionPointToEnd(fn.getBody());
      b.create<KGEN::ReturnOp>(xtorLoc);
      return fn;
    };
    LIT::FuncOp ctorFn = makeXtor(ctorLoc, ctorName, op.getCtor());
    LIT::FuncOp dtorFn = makeXtor(dtorLoc, dtorName, op.getDtor());

    // If the global had a linkage name, make sure it gets renamed.
    StringAttr linkageName = op.getLinkageNameAttr();
    if (linkageName) {
      renamedSymbols.try_emplace(b.getStringAttr(getFlattenedSymbolName(
                                     getFullyResolvedSymbolRef(op))),
                                 linkageName);
    }

    // Replace the `lit.globalvar.decl` operation with a `kgen.global`.
    b.setInsertionPoint(op);
    b.replaceOpWithNewOp<GlobalOp>(
        op, name, op.getType(), b.getI32IntegerAttr(initOrder++),
        getFullyResolvedSymbolRef(ctorFn), getFullyResolvedSymbolRef(dtorFn),
        op.getExportKind());

    for (GlobalRefNode *refdBy : node->refdBy)
      if (++refdBy->numReady == refdBy->numRefs)
        queue.push_back(refdBy);
  }

  if (initOrder != state.size()) {
    return mlir::emitError(
        module.getLoc(),
        "cyclic dependencies between global variables in 'lower-lit' pass");
  }
  return success();
}

/// Add a kgen.link directive that shadows the package's name if the package was
/// precompiled. If this package is a source package, do nothing.
static LogicalResult addPackageLinkDirective(LIT::PackageOp package,
                                             SymbolTable &symtab) {
  // If the package wasn't compiled for anything, it's a source package, so
  // there are no link directives to insert.
  TargetInfoAttr compiledFor = package.getCompiledForAttr();
  if (!compiledFor)
    return success();

  // We have an archive, insert the link directive.
  LIT::PackageArchiveAttr archive = package.getArchiveAttr();
  if (!archive) {
    return package->emitError("expected archive to be provided since the "
                              "package has a target specified");
  }

  OpBuilder b(package.getContext());
  auto linkOp = b.create<PackageLinkOp>(
      package.getLoc(), package.getSymNameAttr(),
      package.getPreElaborationModuleAttr(), compiledFor,
      package.getCompiledEnvAttr(), archive);

  // Insert the link op into the symbol table right where the package was. Don't
  // erase the package op cause we need to do some cleanup still, but we do
  // still want to remove it from the symbol table.
  auto iter = package->getIterator();
  symtab.remove(package);
  symtab.insert(linkOp, iter);

  return success();
}

/// Lower the constructs within the body of a module decl.
static LogicalResult
lowerModuleDecl(Block *moduleBody, SymbolTable &symbolTable,
                DenseMap<StringAttr, StringAttr> &renamedSymbols,
                Block::iterator symTableIt = {},
                const Twine &parentPrefix = {}) {
  bool isTopLevel = symTableIt == Block::iterator();
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    // If we are already in the symbol table, use the the operations iterator.
    auto opSymTableIt = isTopLevel ? op.getIterator() : symTableIt;

    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(&op)
            .Case([&](LIT::FuncOp op) {
              return lowerLITFunc(op, symbolTable, renamedSymbols, opSymTableIt,
                                  parentPrefix);
            })
            .Case([&](StructDeclOp op) {
              return lowerStructDecl(op, symbolTable, renamedSymbols,
                                     opSymTableIt);
            })
            .Case<LIT::FileModuleOp, LIT::PackageOp>([&](auto op) {
              // Lower the constructs within the body.
              Block *fileBody = op.getBody();
              if (failed(lowerModuleDecl(fileBody, symbolTable, renamedSymbols,
                                         opSymTableIt,
                                         parentPrefix + op.getName() + "::")))
                return failure();

              // If the package has already been compiled, insert a kgen.link
              // directive.
              if constexpr (std::is_same_v<decltype(op), LIT::PackageOp>)
                if (failed(addPackageLinkDirective(op, symbolTable)))
                  return failure();

              // Inline the remaining body of the file into the parent.
              op->getBlock()->getOperations().splice(
                  op->getIterator(), fileBody->getOperations(),
                  fileBody->begin(), fileBody->end());
              // Make sure to remove the op from the symbol table if needed.
              if (op->getParentOp() == symbolTable.getOp())
                symbolTable.erase(op);
              else
                op->erase();
              return mlir::success();
            })
            .Case<AliasDeclOp, UnresolvedImportOp, UnresolvedWildcardImportOp>(
                [&](auto op) {
                  op->erase();
                  return mlir::success();
                })
            .Case([&](GlobalOp op) {
              flattenAndRenameSymbol(op, symbolTable, opSymTableIt);
              if (StringAttr linkageName =
                      renamedSymbols.lookup(op.getSymNameAttr()))
                op.setSymNameAttr(linkageName);
              return mlir::success();
            })
            .Case([&](mlir::SymbolOpInterface symbol) {
              flattenAndRenameSymbol(symbol, symbolTable, opSymTableIt);
              return mlir::success();
            })
            .Default(mlir::success());
    if (failed(result))
      return failure();
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
    DenseMap<StringAttr, StringAttr> renamedSymbols;
    if (failed(orderAndLowerGlobalVariables(module, renamedSymbols)))
      return signalPassFailure();
    if (failed(lowerModuleDecl(module.getBody(), symbolTable, renamedSymbols)))
      return signalPassFailure();
    lowerAttributesAndTypes(module, renamedSymbols);
  }
};

} // namespace
