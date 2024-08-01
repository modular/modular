//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace M::DebugInfo;

/// Try to extract a value name from the given source location.
static std::optional<StringRef> getNameFromLoc(llvm::SMRange loc) {
  if (!loc.isValid())
    return std::nullopt;

  StringRef name(loc.Start.getPointer(),
                 loc.End.getPointer() - loc.Start.getPointer());
  name.consume_front("%");
  return name;
}

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

namespace {
/// This class is used to build debug information based on the given IR.
class DebugInfoBuilder {
public:
  DebugInfoBuilder(MLIRContext *context, mlir::AsmParserState &asmState,
                   llvm::SourceMgr &sourceMgr, EmissionKind emissionKind,
                   llvm::dwarf::SourceLanguage debugInfoLanguage)
      : builder(context), dibuilder(context), asmState(asmState),
        sourceMgr(sourceMgr), emissionKind(emissionKind),
        debugInfoLanguage(debugInfoLanguage) {
    // Build the main file descriptor.
    StringRef filename = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())
                             ->getBufferIdentifier();
    fileAttr = buildFile(filename);
  }

  /// Build debug information for the given root operation.
  void build(Operation *op);

private:
  DIBuilder::ScopeGuard buildDebugInfo(mlir::FunctionOpInterface op);
  DIBuilder::ScopeGuard buildDebugInfo(SubprogramScoped sp);
  void buildDebugInfo(Region *region, bool pushLexicalBlock);
  void buildDebugInfo(Block *block, bool isFunctionEntryBlock = false);
  void buildDebugInfo(Operation *op);

  /// Build debug information of a local variable for the given value.
  DILocalVariableAttr buildLocalVariable(const Twine &name, unsigned line,
                                         Value value, unsigned argNo = 0);

  /// Build a file attribute for the given filename.
  DIFileAttr buildFile(StringRef filename);

  /// Extract the line and column of the given location.
  std::pair<unsigned, unsigned> extractLineColumn(Operation *op);
  std::pair<unsigned, unsigned> extractLineColumn(Block *block);
  std::pair<unsigned, unsigned> extractLineColumn(llvm::SMRange loc);
  template <typename T>
  unsigned extractLine(T loc) {
    return extractLineColumn(loc).first;
  }

  /// A builder used to simplify attribute/type creation.
  Builder builder;

  /// A builder used for generating the debug info constructs.
  DIBuilder dibuilder;

  /// The asm parser state used to resolve the source information for the given
  /// IR.
  mlir::AsmParserState &asmState;

  /// A file attribute for the source buffer of the current IR.
  DIFileAttr fileAttr;

  /// The type converters used for building debug information from MLIR types.
  DebugInfoTypeConverter typeConverter;

  /// The source manager that owns the state within `asmState`.
  llvm::SourceMgr &sourceMgr;

  /// The kind of debug information emission.
  EmissionKind emissionKind;

  /// The language to specify in the debug info.
  llvm::dwarf::SourceLanguage debugInfoLanguage;
};
} // namespace

void DebugInfoBuilder::build(Operation *op) {
  if (emissionKind == EmissionKind::None)
    return;
  auto fileGuard = dibuilder.pushScopeGuard(fileAttr);

  // Attach compile unit information to `op`.
  dibuilder.initializeCompileUnit(debugInfoLanguage, fileAttr,
                                  /*producer=*/"MLIR",
                                  /*isOptimized=*/true, EmissionKind::Full);

  // Populate debug information for operations nested under `op`.
  op->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op)) {
      auto scopeGuard = buildDebugInfo(funcOp);
      if (funcOp.isExternal())
        return WalkResult::skip();

      // Don't recurse if we only want line table information. We really just
      // need a subprogram for that.
      if (emissionKind == EmissionKind::LineTablesOnly)
        return WalkResult::skip();

      Region &body = funcOp.getFunctionBody();
      if (body.empty())
        return WalkResult::skip();

      buildDebugInfo(&body.front(), /*isFunctionEntryBlock=*/true);
      for (Block &block : llvm::drop_begin(body))
        buildDebugInfo(&block);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

DIBuilder::ScopeGuard
DebugInfoBuilder::buildDebugInfo(mlir::FunctionOpInterface op) {
  // Build debug information for the type of the function.
  SmallVector<DIType> resultTypes, argumentTypes;
  for (Type type : op.getResultTypes())
    resultTypes.push_back(typeConverter.convertDebugType(type));
  for (Type type : op.getArgumentTypes())
    argumentTypes.push_back(typeConverter.convertDebugType(type));
  auto subroutineType =
      builder.getType<DISubroutineType>(argumentTypes, resultTypes);

  // Build the subprogram for this function.
  unsigned line = extractLine(op);
  StringAttr name = op.getNameAttr();
  auto spFlags = SubprogramFlags::Optimized;
  if (!op.isExternal()) {
    // TODO: Add enum support for `|=`.
    spFlags = SubprogramFlags::Optimized | SubprogramFlags::Definition;
  }
  auto spGuard =
      dibuilder.pushSubprogram(SourceNameAttr::get(name), name, fileAttr, line,
                               line, spFlags, subroutineType);
  op->setLoc(dibuilder.createScopedLoc(op->getLoc()));
  return spGuard;
}

DIBuilder::ScopeGuard DebugInfoBuilder::buildDebugInfo(SubprogramScoped sp) {
  // The CallLoc is the instantiated location of the inlined subprogram.
  Location callLoc = dibuilder.createScopedLoc(sp->getLoc());

  SmallVector<DIType> resultTypes, argumentTypes;
  for (Type type : sp.getBodyRegion().getArgumentTypes())
    resultTypes.push_back(typeConverter.convertDebugType(type));
  for (Type type : sp->getResultTypes())
    argumentTypes.push_back(typeConverter.convertDebugType(type));
  auto subroutineType =
      builder.getType<DISubroutineType>(argumentTypes, resultTypes);

  StringAttr name = sp->getName().getIdentifier();
  unsigned line = extractLine(sp);
  SubprogramFlags spFlags =
      SubprogramFlags::Optimized | SubprogramFlags::Definition;
  DIBuilder::ScopeGuard spGuard =
      dibuilder.pushSubprogram(SourceNameAttr::get(name), name, fileAttr, line,
                               line, spFlags, subroutineType);
  sp->setLoc(dibuilder.createScopedLoc(sp->getLoc()));

  if (auto isp = dyn_cast<InlinedSubprogramScoped>(sp.getOperation()))
    isp.setCallLocAttr(callLoc);

  return spGuard;
}

void DebugInfoBuilder::buildDebugInfo(Region *region, bool pushLexicalBlock) {
  if (region->empty())
    return;

  DIBuilder::ScopeGuard scopeGuard;
  if (pushLexicalBlock) {
    auto [line, column] = extractLineColumn(&region->front());
    scopeGuard = dibuilder.pushNestedLexicalBlock(fileAttr, line, column);
  }

  // Recursively build debug information for all blocks within the region.
  for (Block &block : *region)
    buildDebugInfo(&block);
}

void DebugInfoBuilder::buildDebugInfo(Block *block, bool isFunctionEntryBlock) {
  // Add debug information for the arguments.
  if (const mlir::AsmParserState::BlockDefinition *blockDef =
          asmState.getBlockDef(block)) {
    OpBuilder blockBuilder = OpBuilder::atBlockBegin(block);
    for (auto [index, arg] : llvm::enumerate(block->getArguments())) {
      unsigned argNo = isFunctionEntryBlock ? (index + 1) : 0;
      auto &argInfo = blockDef->arguments[index];

      // Try to extract a name for this argument.
      // TODO: Support unnamed arguments?
      std::optional<StringRef> name = getNameFromLoc(argInfo.loc);
      if (!name)
        continue;
      blockBuilder.create<ValueOp>(
          arg.getLoc(), arg,
          buildLocalVariable(*name, extractLine(argInfo.loc), arg, argNo));
    }
  } else {
    // TODO: Add artificial information for arguments of implicit blocks?
  }

  // Add debug information for the operations.
  for (Operation &op : *block)
    buildDebugInfo(&op);
}

void DebugInfoBuilder::buildDebugInfo(Operation *op) {
  if (auto sp = dyn_cast<SubprogramScoped>(op)) {
    DIBuilder::ScopeGuard scopeGuard = buildDebugInfo(sp);
    buildDebugInfo(&sp.getBodyRegion(), /*pushLexicalBlock=*/false);
    return;
  }

  op->setLoc(dibuilder.createScopedLoc(op->getLoc()));

  // Traverse into regions of this operation.
  for (auto &region : op->getRegions())
    buildDebugInfo(&region, /*pushLexicalBlock=*/true);

  // Check for results to this operation.
  auto opResults = op->getResults();
  if (opResults.empty())
    return;
  OpBuilder blockBuilder(op->getContext());
  blockBuilder.setInsertionPointAfter(op);

  // If we have a definition for this operation, use it to extract the names.
  if (const auto *opDef = asmState.getOpDef(op)) {
    unsigned numResultGroups = opDef->resultGroups.size();
    for (auto [index, resultGroup] : llvm::enumerate(opDef->resultGroups)) {
      // TODO: Support unnamed results?
      std::optional<StringRef> name =
          getNameFromLoc(resultGroup.definition.loc);
      if (!name)
        continue;
      unsigned line = extractLine(resultGroup.definition.loc);

      int startIt = resultGroup.startIndex;
      int nextIt = (index == (numResultGroups - 1))
                       ? opResults.size()
                       : opDef->resultGroups[index + 1].startIndex;
      bool isSingleResult = (startIt + 1) == nextIt;
      for (int it : llvm::seq(startIt, nextIt)) {
        blockBuilder.create<ValueOp>(
            op->getLoc(), opResults[it],
            buildLocalVariable(isSingleResult ? *name
                                              : (*name + "#" + Twine(it)),
                               line, opResults[it]));
      }
    }
    return;
  }

  // Otherwise, this operation wasn't defined in the assembly and is implicit
  // somewhere.
  // TODO: We could try to display these, but there isn't a guarantee that the
  // operation has an actual debug location that is useful to us.
}

DILocalVariableAttr DebugInfoBuilder::buildLocalVariable(const Twine &name,
                                                         unsigned line,
                                                         Value value,
                                                         unsigned argNo) {
  return dibuilder.createLocalVariable(
      name.str(), fileAttr, line, argNo, /*alignInBits=*/0,
      typeConverter.convertDebugType(value.getType()));
}

DIFileAttr DebugInfoBuilder::buildFile(StringRef filename) {
  SmallString<256> currentWorkingDir;
  llvm::sys::fs::current_path(currentWorkingDir);

  StringRef directory = currentWorkingDir;
  SmallString<128> dirBuf, fileBuf;
  if (llvm::sys::path::is_absolute(filename)) {
    // Strip the common prefix (if it is more than just "/") from current
    // directory and FileName for a more space-efficient encoding.
    auto fileIt = llvm::sys::path::begin(filename);
    auto fileE = llvm::sys::path::end(filename);
    auto curDirIt = llvm::sys::path::begin(directory);
    auto curDirE = llvm::sys::path::end(directory);
    for (; curDirIt != curDirE && *curDirIt == *fileIt; ++curDirIt, ++fileIt)
      llvm::sys::path::append(dirBuf, *curDirIt);
    if (std::distance(llvm::sys::path::begin(directory), curDirIt) == 1) {
      // Don't strip the common prefix if it is only the root "/" since that
      // would make diagnostic locations confusing.
      directory = StringRef();
    } else {
      for (; fileIt != fileE; ++fileIt)
        llvm::sys::path::append(fileBuf, *fileIt);
      directory = dirBuf;
      filename = fileBuf;
    }
  }
  return dibuilder.createFile(filename, directory);
}

std::pair<unsigned, unsigned>
DebugInfoBuilder::extractLineColumn(Operation *op) {
  if (auto *opDef = asmState.getOpDef(op))
    return extractLineColumn(opDef->loc);
  return {0, 0};
}

std::pair<unsigned, unsigned>
DebugInfoBuilder::extractLineColumn(Block *block) {
  if (auto *blockDef = asmState.getBlockDef(block))
    return extractLineColumn(blockDef->definition.loc);
  return extractLineColumn(block->getParentOp());
}

std::pair<unsigned, unsigned>
DebugInfoBuilder::extractLineColumn(llvm::SMRange loc) {
  if (unsigned locBufId = sourceMgr.FindBufferContainingLoc(loc.Start))
    return sourceMgr.getLineAndColumn(loc.Start, locBufId);
  return {0, 0};
}

//===----------------------------------------------------------------------===//
// Parser Entry
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> DebugInfo::parseSourceFileWithDebugInfo(
    llvm::SourceMgr &sourceMgr, const mlir::ParserConfig &config,
    EmissionKind emissionKind, llvm::dwarf::SourceLanguage debugInfoLanguage) {
  const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // If the buffer is bytecode we can't attach debug info directly on parse, so
  // just parse it as normal and then snapshot afterwards.
  if (mlir::isBytecode(*sourceBuf)) {
    OwningOpRef<ModuleOp> module =
        mlir::parseSourceFile<ModuleOp>(sourceMgr, config);
    if (!module || failed(snapshotDebugInfo(*module, /*filename=*/"",
                                            emissionKind, debugInfoLanguage)))
      return nullptr;
    return module;
  }

  // Otherwise, we need to parse the file and attach debug info.
  Block block;
  mlir::AsmParserState parserState;
  if (failed(mlir::parseAsmSourceFile(sourceMgr, &block, config, &parserState)))
    return nullptr;

  // Construct the module op from the parsed block.
  Location parserLoc =
      FileLineColLoc::get(config.getContext(), sourceBuf->getBufferIdentifier(),
                          /*line=*/0, /*column=*/0);
  auto moduleOp =
      mlir::detail::constructContainerOpForParserIfNecessary<ModuleOp>(
          &block, config.getContext(), parserLoc);

  // Attach debug info to the module op.
  DebugInfoBuilder builder(config.getContext(), parserState, sourceMgr,
                           emissionKind, debugInfoLanguage);
  builder.build(*moduleOp);
  return moduleOp;
}

//===----------------------------------------------------------------------===//
// Snapshot Entry
//===----------------------------------------------------------------------===//

LogicalResult
DebugInfo::snapshotDebugInfo(Operation *op, StringRef filename,
                             EmissionKind emissionKind,
                             llvm::dwarf::SourceLanguage debugInfoLanguage) {
  // Kill any pre-existing debug info operations.
  op->walk([](Operation *op) {
    if (llvm::isa_and_present<DebugInfoDialect>(op->getDialect()))
      op->erase();
  });

  // If a filename wasn't provided, then generate one.
  SmallString<32> filepath(filename);
  if (filepath.empty()) {
    if (std::error_code error = llvm::sys::fs::createTemporaryFile(
            "mlir_snapshot", "tmp.mlir", filepath)) {
      return op->emitError()
             << "failed to generate temporary file for location snapshot: "
             << error.message();
    }
  }

  // Output the IR to the filepath.
  std::string error;
  {
    std::unique_ptr<llvm::ToolOutputFile> outputFile =
        mlir::openOutputFile(filepath, &error);
    if (!outputFile)
      return op->emitError() << error;
    op->print(outputFile->os(), mlir::OpPrintingFlags().enableDebugInfo(false));
    outputFile->keep();
  }

  // Re-open the file into a new source manager.
  auto newFile = mlir::openInputFile(filepath, &error);
  if (!newFile)
    return op->emitError() << error;
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(newFile), llvm::SMLoc());

  // Parse the operation back in, collecting source information about the IR
  // as we parse.
  Block block;
  mlir::AsmParserState parserState;
  if (failed(mlir::parseAsmSourceFile(sourceMgr, &block, op->getContext(),
                                      &parserState)))
    return failure();

  // Attach debug info to the newly parsed operation.
  Operation *parsedOp = &block.front();
  DebugInfoBuilder builder(op->getContext(), parserState, sourceMgr,
                           emissionKind, debugInfoLanguage);
  builder.build(parsedOp);

  // Replace the current operation with the reconstructed parser version.
  op->setLoc(parsedOp->getLoc());
  for (const auto &[opRegion, parsedRegion] :
       llvm::zip(op->getRegions(), parsedOp->getRegions()))
    opRegion.takeBody(parsedRegion);
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
#define GEN_PASS_DEF_DEBUGINFOSNAPSHOT
#include "Support/DebugInfoDialect/Transforms/Transforms.h.inc"
} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct DebugInfoSnapshot
    : public impl::DebugInfoSnapshotBase<DebugInfoSnapshot> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void DebugInfoSnapshot::runOnOperation() {
  if (failed(snapshotDebugInfo(getOperation(), filename, emissionKind,
                               debugInfoLanguage)))
    signalPassFailure();
}
