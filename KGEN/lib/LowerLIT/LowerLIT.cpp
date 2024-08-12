//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
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
  return SymbolRefAttr::get(ref.getContext(), getFlattenedSymbolName(ref));
}

/// This processes a `lit.func` and returns the param declarations for the
/// normal input parameters, ignoring the lifetime parameters.
static ArrayRef<ParamDeclAttr> extractImplicitLifetimeParams(LIT::FuncOp func) {
  size_t numImplicitLifetimes =
      func.getSignature().getNumImplicitLifetimeDecls();
  return func.getInputParams().drop_back(numImplicitLifetimes);
}

//===----------------------------------------------------------------------===//
// Op Lowering
//===----------------------------------------------------------------------===//

namespace {
struct LITLowerer {
  /// Given a function, check to see if it is a top-level function.  If not,
  /// lower it to a ParamDeclareRegionOp.
  void lowerNestedFunction(LIT::FuncOp func);
  /// Lower LIT dialect operations in a function body.
  void lowerLITOps(LIT::FuncOp func);
  /// Lower a lit.func to kgen.generator.
  LogicalResult lowerLITFunc(LIT::FuncOp func, Block::iterator symTableIt,
                             const Twine &parentPrefix);
  LogicalResult lowerLITFunc(LIT::FuncOp func, Block::iterator symTableIt,
                             const Twine &parentPrefix,
                             ArrayRef<ParamDeclAttr> parentInputParams,
                             ArrayRef<bool> parentVariadicMask);
  /// Lower nested structures in lit.struct.decl away.
  LogicalResult lowerStructDecl(StructDeclOp structDecl,
                                Block::iterator symTableIt);
  /// Lower the constructs within the body of a module decl.
  LogicalResult lowerModuleDecl(Block *moduleBody,
                                Block::iterator symTableIt = {},
                                const Twine &parentPrefix = {});

  SymbolTable &symbolTable;
  DenseMap<StringAttr, StringAttr> &renamedSymbols;
};
} // namespace

void LITLowerer::lowerLITOps(LIT::FuncOp func) {
  func.getBodyRegion().walk([&](Operation *op) {
    // Lower any aliases within the function body to param declare.
    mlir::IRRewriter b{OpBuilder(op)};
    if (AliasDeclOp alias = dyn_cast<AliasDeclOp>(op)) {
      b.replaceOpWithNewOp<ParamDeclareOp>(
          alias, TypeRange(), alias.getParamDecl(), alias.getValue());
    } else if (auto lifetimeStart = dyn_cast<VarLifetimeStartOp>(op)) {
      auto arg = lifetimeStart.getArg();
      b.replaceOpWithNewOp<POP::StackAllocLifetimeStartOp>(
          op, arg.getDefiningOp()->getOperand(0));
    } else if (auto lifetimeEnd = dyn_cast<VarLifetimeEndOp>(op)) {
      auto arg = lifetimeEnd.getArg();
      b.replaceOpWithNewOp<POP::StackAllocLifetimeEndOp>(
          op, arg.getDefiningOp()->getOperand(0));
    } else if (isa<OwnershipUseOp, OwnershipUseLifetimeOp,
                   OwnershipMarkInitializedOp, OwnershipMarkDestroyedOp,
                   OwnershipMarkConsumedOp, OwnershipDefLValueOp,
                   UnresolvedImportOp, UnresolvedWildcardImportOp>(op)) {
      // lit.ownership.* are used internally by the
      // frontend and ownership lowering, but is not needed after that.
      op->erase();
    } else if (isa<TransferRegOwnershipOp>(op)) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    } else if (auto transfer = dyn_cast<TransferMemOwnershipOp>(op)) {
      b.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          transfer, ArrayRef<Type>(transfer.getType()), transfer.getOperand());
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
      auto allocOp = b.create<POP::StackAllocationOp>(
          varDecl.getLoc(), varDecl.getType().getAsPointerType(), 1,
          /*markedLifetimes=*/true);

      // Replace !lit.ref result type with a cast from the pointer.  This will
      // get squashed by LowerLITTypes.
      b.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          varDecl, ArrayRef<Type>(varDecl.getType()), allocOp.getResult());
    } else if (auto returnOp = dyn_cast<ErrorReturnOp>(op)) {
      b.replaceOpWithNewOp<KGEN::ReturnOp>(returnOp, returnOp.getResult());
    } else if (auto elifOp = dyn_cast<HLCF::ElifOp>(op)) {
      HLCF::replaceElifWithIfOps(elifOp);
    } else if (auto globalRefOp = dyn_cast<GlobalVarRefOp>(op)) {
      Value newAddr = b.create<GlobalAddressOp>(
          op->getLoc(), globalRefOp.getType().getAsPointerType(),
          globalRefOp.getGlobal());

      // Replace !lit.ref result type with a cast from the pointer.  This will
      // get squashed by LowerLITTypes.
      b.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
          globalRefOp, ArrayRef<Type>(globalRefOp.getType()), newAddr);

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

LogicalResult LITLowerer::lowerLITFunc(LIT::FuncOp func,
                                       Block::iterator symTableIt,
                                       const Twine &parentPrefix) {
  return lowerLITFunc(func, symTableIt, parentPrefix, /*parentInputParams=*/{},
                      /*parentVariadicMask=*/{});
}

LogicalResult
LITLowerer::lowerLITFunc(LIT::FuncOp func, Block::iterator symTableIt,
                         const Twine &parentPrefix,
                         ArrayRef<ParamDeclAttr> parentInputParams,
                         ArrayRef<bool> parentVariadicMask) {
  // Update the function name, incorporating the parent prefix.
  if (!parentPrefix.isTriviallyEmpty()) {
    StringAttr newName = flattenAndRenameSymbol(func, symbolTable, symTableIt);

    // If this function has a subprogram attached, update its information to
    // account for the new name.
    DebugInfo::updateSubprogram(func, newName);
  }

  lowerLITOps(func);

  OpBuilder b(func->getContext());
  LITSignatureType signature = func.getSignature();

  ArrayRef<ParamDeclAttr> genParams = extractImplicitLifetimeParams(func);

  // Prepend the parameters from the parent decl if present.
  SmallVector<ParamDeclAttr> inputParams;
  if (!parentInputParams.empty()) {
    // Concat the parent and generator input parameter decls.
    llvm::append_range(inputParams, parentInputParams);
    // Offset index references within the current signature to make room.
    // Remap parent input parameter references to indices.
    signature = LITSignatureType::prependParams(signature, parentInputParams,
                                                parentVariadicMask);
  }
  llvm::append_range(inputParams, genParams);

  Operation *result;
  // If the function has an alias name, rename it.
  if (StringAttr newName = func.getLinkageNameAttr()) {
    renamedSymbols[func.getSymNameAttr()] = newName;
    func.setSymName(newName);
  }

  auto inputParamsArr = ParamDeclArrayAttr::get(b.getContext(), inputParams);
  auto resParamsArr = ParamDeclArrayAttr::get(b.getContext(), {});
  auto sigAttr = TypeAttr::get(signature);

  // Directly lower since these operations are exactly identical right now.
  OperationState state(func.getLoc(), GeneratorOp::getOperationName());
  GeneratorOp::build(b, state, func.getSymNameAttr(), sigAttr,
                     func.getFunctionTypeAttr(), inputParamsArr, resParamsArr,
                     func.getDecoratorsAttr(), func.getInlineLevelAttr(),
                     func.getExportKindAttr(), func.getLLVMMetadata());

  for (const NamedAttribute &attr : func->getDialectAttrs())
    state.attributes.push_back(attr);

  auto newFunc = cast<GeneratorOp>(b.create(state));
  result = newFunc;

  // Move over the body.
  newFunc.getBodyRegion().takeBody(func.getBodyRegion());

  // Move over the symbol, and we're done.
  Block::iterator genIter = func->getIterator();
  symbolTable.remove(func);
  symbolTable.insert(result, genIter);
  func.erase();

  return success();
}

void LITLowerer::lowerNestedFunction(LIT::FuncOp func) {
  // Process a nested function by lowering it straight to a
  // `kgen.param.declare.region`. Nested functions are denoted with an
  // parameter declaration on the function declaration.
  ParamDeclAttr decl = func.getParamDeclAttr();
  assert(decl && "expected nested function to declare a parameter");

  ImplicitLocOpBuilder b(func.getLoc(), func);

  // The new param.declare.region will drop implicit lifetimes.
  ArrayRef<ParamDeclAttr> inputParams = extractImplicitLifetimeParams(func);

  auto region = b.create<ParamDeclareRegionOp>(
      decl, func.getSignature(), func.getFunctionType(), inputParams,
      func.getResultParams(), /*isolated=*/false, func.getInlineLevel());
  region.getBodyRegion().takeBody(func.getBodyRegion());
  func.erase();
}

LogicalResult LITLowerer::lowerStructDecl(StructDeclOp structDecl,
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
    SmallVector<bool> variadicMask = llvm::map_to_vector(
        structDecl.getSignature().getParamListAttrs().getPogs(),
        [](PogMetadataAttr pogAttr) { return pogAttr.isVariadic(); });
    if (failed(lowerLITFunc(func, structDecl->getIterator(),
                            structName.getValue() + "::",
                            structDecl.getInputParams(), variadicMask)))
      return failure();
  }
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
  auto linkOp = b.create<PackageLinkOp>(
      package.getLoc(), package.getSymNameAttr(),
      package.getPostParseModuleAttr(), package.getDependenciesAttr());

  // Insert the link op into the symbol table right where the package was. Don't
  // erase the package op cause we need to do some cleanup still, but we do
  // still want to remove it from the symbol table.
  auto iter = package->getIterator();
  symtab.remove(package);
  symtab.insert(linkOp, iter);

  return success();
}

LogicalResult LITLowerer::lowerModuleDecl(Block *moduleBody,
                                          Block::iterator symTableIt,
                                          const Twine &parentPrefix) {
  bool isTopLevel = symTableIt == Block::iterator();
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    // If we are already in the symbol table, use the the operations iterator.
    auto opSymTableIt = isTopLevel ? op.getIterator() : symTableIt;

    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(&op)
            .Case([&](LIT::FuncOp op) {
              return lowerLITFunc(op, opSymTableIt, parentPrefix);
            })
            .Case([&](StructDeclOp op) {
              return lowerStructDecl(op, opSymTableIt);
            })
            .Case<LIT::FileModuleOp, LIT::PackageOp>([&](auto op) {
              // Make sure to remove the op from the symbol table if needed.
              if (op->getParentOp() == symbolTable.getOp())
                symbolTable.remove(op);

              // Lower the constructs within the body.
              Block *fileBody = op.getBody();
              if (failed(lowerModuleDecl(fileBody, opSymTableIt,
                                         parentPrefix + op.getName() + "::")))
                return failure();

              // If the package has already been compiled, insert a link
              // directive.
              if constexpr (std::is_same_v<decltype(op), LIT::PackageOp>)
                if (failed(addPackageLinkDirective(op, symbolTable)))
                  return failure();

              // Inline the remaining body of the file into the parent.
              op->getBlock()->getOperations().splice(
                  op->getIterator(), fileBody->getOperations(),
                  fileBody->begin(), fileBody->end());
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
// Type lowering
//===----------------------------------------------------------------------===//

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

  // Remove signature metadata.
  replacer.addReplacement([](SignatureType sig) {
    return SignatureType::get(sig.getValues(), sig.getInputParamTypes(),
                              sig.getResultParamTypes(),
                              sig.getArgConventions(), sig.getFnEffects());
  });

  replacer.addReplacement([](TypedAttr attr) -> TypedAttr {
    if (auto type = dyn_cast<LifetimeType>(attr.getType()))
      return LifetimeAttr::get(type);
    return attr;
  });

  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);
}

//===----------------------------------------------------------------------===//
// Global Variables
//===----------------------------------------------------------------------===//

/// Global variables have initializers that can reference other global
/// variables. This function will pass over all global variable declarations and
/// determine an initialization order based on the reference graph between
/// global variables in their initializers. It is not possible for the parser to
/// generate global variables that reference each other in a cycle.
static LogicalResult
orderAndLowerGlobalVariables(ModuleOp module,
                             DenseMap<StringAttr, StringAttr> &renamedSymbols,
                             llvm::dwarf::SourceLanguage debugInfoLanguage) {
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

      DebugInfo::DIBuilder dib(ctx);
      dib.initializeCompileUnit(debugInfoLanguage, fileAttr, "kgen",
                                /*isOptimized=*/true,
                                DebugInfo::EmissionKind::Full);

      // We set the scoped location for the outlined methods.
      auto spType = DebugInfo::DISubroutineType::get(ctx, {}, {});
      auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
      DebugInfo::DIBuilder::ScopeGuard guard = dib.pushScopeGuard(fileAttr);
      auto getXtorLoc = [&](StringAttr xtorName) {
        guard = dib.pushSubprogram(
            DebugInfo::SourceNameAttr::get(xtorName), xtorName, fileAttr,
            fileLoc.getLine(), fileLoc.getLine(),
            DebugInfo::SubprogramFlags::Definition, spType);
        return dib.createScopedLoc(fileLoc);
      };
      ctorLoc = getXtorLoc(ctorName);
      dtorLoc = getXtorLoc(dtorName);
    }

    // Outline the constructor and destructor into functions.
    auto sig = LITSignatureType::get(b.getContext(), /*inputs=*/TypeRange{},
                                     /*results=*/TypeRange{},
                                     /*numImplicitLifetimeDecls=*/0);
    auto makeXtor = [&](Location xtorLoc, StringAttr xtorName, Region &body) {
      b.setInsertionPoint(op);
      auto fn = b.create<LIT::FuncOp>(xtorLoc, xtorName, StringAttr(), sig);
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
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();

    DenseMap<StringAttr, StringAttr> renamedSymbols;
    if (failed(orderAndLowerGlobalVariables(
            module, renamedSymbols,
            static_cast<llvm::dwarf::SourceLanguage>(
                debugInfoLanguage.getValue()))))
      return signalPassFailure();

    LITLowerer lowerer{analysis.getTopLevelSymbolTable(), renamedSymbols};
    if (failed(lowerer.lowerModuleDecl(module.getBody())))
      return signalPassFailure();
    lowerAttributesAndTypes(module, renamedSymbols);
  }
};

} // namespace
