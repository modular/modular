//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace KGEN;

// FIXME: This entire file is a workaround debuginfo generation in the front-end
// being broken. The reasons it is broken surrounds various issues involving
// parameters. This synthetic debuginfo generation runs after elaboration, where
// no such problems are possible.

//===----------------------------------------------------------------------===//
// synthesizeDebugInfo
//===----------------------------------------------------------------------===//

/// In general, each region corresponds to a Mojo lexical block.
static LogicalResult visitLexicalRegion(Region &region,
                                        DebugInfo::DIBuilder &dib) {
  for (Operation &op : region.front()) {
    op.setLoc(dib.createScopedLoc(op.getLoc()));
    for (Region &region : op.getRegions()) {
      auto fileLoc = op.getLoc()->findInstanceOf<FileLineColLoc>();
      if (!fileLoc)
        return mlir::emitError(op.getLoc()) << "did not find a FileLineColLoc";
      DebugInfo::DIBuilder::ScopeGuard guard = dib.pushNestedLexicalBlock(
          dib.createFile(fileLoc), fileLoc.getLine(), fileLoc.getColumn());
      if (failed(visitLexicalRegion(region, dib)))
        return failure();
    }
  }
  return success();
}

/// Synthesize fake debug info for the module given concrete IR using knowledge
/// of how the IR is lowered from Mojo. In doing so, we try to dig out
/// FileLineColLoc. If the file came from the front-end, this should be no
/// issue. Otherwise, the pass fails.
static LogicalResult
synthesizeDebugInfo(ModuleOp module,
                    llvm::dwarf::SourceLanguage debugInfoLanguage) {
  MLIRContext *ctx = module.getContext();
  DebugInfo::DIBuilder dib(ctx);
  // Attempt to dig out a file from the module location.
  auto fileLoc = module.getLoc()->findInstanceOf<FileLineColLoc>();
  if (!fileLoc)
    return mlir::emitError(module.getLoc()) << "did not find a FileLineColLoc";

  dib.initializeCompileUnit(debugInfoLanguage, dib.createFile(fileLoc), "kgen",
                            /*isOptimized=*/true,
                            DebugInfo::EmissionKind::LineTablesOnly);
  DebugInfo::DIBuilder::ScopeGuard moduleGuard =
      dib.pushFile(fileLoc.getFilename(), "/");

  module->setLoc(dib.createScopedLoc(module.getLoc()));

  // TODO(14025): Add proper conversions for KGEN and POP types.
  DebugInfo::DebugInfoTypeConverter tc;
  auto toDIType = [&](Type type) -> DebugInfo::DIType {
    return DebugInfo::DIBasicUIntType::get(ctx, "unknown", 8, 8);
  };

  // TODO: Parallelize this.
  for (auto func : module.getOps<FuncOp>()) {
    auto fileLoc = func.getLoc()->findInstanceOf<FileLineColLoc>();
    if (!fileLoc)
      return mlir::emitError(func.getLoc()) << "did not find a FileLineColLoc";
    // Generate the subprogram type.
    auto spType = DebugInfo::DISubroutineType::get(
        ctx, llvm::map_to_vector(func.getArgumentTypes(), toDIType),
        llvm::map_to_vector(func.getResultTypes(), toDIType));

    DebugInfo::DIBuilder::ScopeGuard fileGuard =
        dib.pushFile(fileLoc.getFilename(), "/");

    // Attempt to determine an un-mangled name.
    StringAttr name = func.getSymNameAttr();
    FailureOr<LIT::MangledSymbol> symbol =
        LIT::MangledSymbol::demangle(name, /*parseSignature=*/false);
    DebugInfo::SourceNameAttr sourceName;
    if (failed(symbol)) {
      sourceName = DebugInfo::SourceNameAttr::get(name);
    } else {
      // Mangle the namespaces back.
      for (StringAttr name : symbol->moduleNames) {
        sourceName = DebugInfo::SourceNameAttr::get(
            name, DebugInfo::SourceNameKind::Module, sourceName);
      }
      for (StringAttr name : symbol->structNames) {
        sourceName = DebugInfo::SourceNameAttr::get(
            name, DebugInfo::SourceNameKind::Struct, sourceName);
      }
      sourceName = DebugInfo::SourceNameAttr::get(
          symbol->symName, DebugInfo::SourceNameKind::Fn, sourceName);
    }

    DebugInfo::DIBuilder::ScopeGuard guard =
        dib.pushSubprogram(sourceName, name, dib.createFile(fileLoc),
                           fileLoc.getLine(), fileLoc.getLine(),
                           DebugInfo::SubprogramFlags::Definition |
                               DebugInfo::SubprogramFlags::Optimized,
                           spType);
    func->setLoc(dib.createScopedLoc(fileLoc));
    if (failed(visitLexicalRegion(func.getBodyRegion(), dib)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_SYNTHESIZEDEBUGINFO
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct SynthesizeDebugInfoPass
    : public impl::SynthesizeDebugInfoBase<SynthesizeDebugInfoPass> {
  using SynthesizeDebugInfoBase::SynthesizeDebugInfoBase;

  void runOnOperation() override {
    if (failed(synthesizeDebugInfo(getOperation(),
                                   static_cast<llvm::dwarf::SourceLanguage>(
                                       debugInfoLanguage.getValue()))))
      return signalPassFailure();
  }
};
} // namespace
