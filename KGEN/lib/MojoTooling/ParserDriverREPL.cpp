//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoLexer/Lexer.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/CodeComplete.h"
#include "KGEN/POPDialect/POPOps.h"
#include "ParserDriverImpl.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// MojoParserContext::REPLLocMapper
//===----------------------------------------------------------------------===//

/// This class provides support for mapping locations between an input REPL
/// expression and the wrapped expression that is actually parsed.
class MojoParserContext::REPLLocMapper::ExprLocMapper {
public:
  ExprLocMapper(StringRef inputExpr)
      : inputExpr(inputExpr), inputToWrappedMap(allocator),
        wrappedToInputMap(allocator) {}

  /// Set the expressions mapped in this mapper.
  void setWrappedExpr(StringRef exprText) { wrappedExpr = exprText; }

  /// Map a substring of the REPL input expression to the same corresponding
  /// substring within the wrapped expression.
  void addMapping(StringRef inputExprSplice, unsigned wrappedExprOffset) {
    // Insert a mapping from input to wrapped expression.
    unsigned inputExprOffset = inputExprSplice.data() - inputExpr.data();
    inputToWrappedMap.insert(inputExprOffset,
                             inputExprOffset + inputExprSplice.size(),
                             wrappedExprOffset);

    // Insert a reverse mapping from wrapped to input expression.
    wrappedToInputMap.insert(wrappedExprOffset,
                             wrappedExprOffset + inputExprSplice.size(),
                             inputExprOffset);
  }

  /// Map the given location in the input expression to the wrapped expression.
  /// Returns an invalid location if the location is not mapped.
  llvm::SMLoc mapLocation(llvm::SMLoc loc) const {
    auto mapImpl = [&](const char *locBufferStart, const char *newBufferStart,
                       const MapT &map) {
      unsigned locOffset = loc.getPointer() - locBufferStart;

      auto it = map.find(locOffset);
      if (!it.valid() || locOffset < it.start())
        return llvm::SMLoc();
      return llvm::SMLoc::getFromPointer(newBufferStart + it.value() +
                                         (locOffset - it.start()));
    };

    // Check if the location is within the input or wrapped expression.
    if (loc.getPointer() >= inputExpr.data() &&
        loc.getPointer() < inputExpr.end()) {
      return mapImpl(inputExpr.data(), wrappedExpr.data(), wrappedToInputMap);
    }
    if (loc.getPointer() >= wrappedExpr.data() &&
        loc.getPointer() < wrappedExpr.end()) {
      return mapImpl(wrappedExpr.data(), inputExpr.data(), wrappedToInputMap);
    }
    return llvm::SMLoc();
  }

private:
  using MapT = llvm::IntervalMap<
      unsigned, unsigned,
      llvm::IntervalMapImpl::NodeSizer<unsigned, StringRef>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<unsigned>>;
  MapT::Allocator allocator;

  /// The buffer for the input expression.
  StringRef inputExpr;
  MapT inputToWrappedMap;

  /// The buffer for the wrapped expression.
  StringRef wrappedExpr;
  MapT wrappedToInputMap;
};

MojoParserContext::REPLLocMapper::REPLLocMapper(llvm::SourceMgr &sourceMgr)
    : sourceMgr(sourceMgr) {}
MojoParserContext::REPLLocMapper::~REPLLocMapper() = default;

llvm::SMLoc
MojoParserContext::REPLLocMapper::mapLocation(llvm::SMLoc loc) const {
  for (ExprLocMapper &mapper : llvm::make_pointee_range(exprMappers))
    if (llvm::SMLoc newLoc = mapper.mapLocation(loc); newLoc.isValid())
      return newLoc;
  return llvm::SMLoc();
}

llvm::SMDiagnostic MojoParserContext::REPLLocMapper::mapDiagnostic(
    const llvm::SMDiagnostic &diag) {
  // Check if the diagnostic is using location information from the wrapped
  // expression.
  llvm::SMLoc newLoc = mapLocation(diag.getLoc());
  if (!newLoc.isValid())
    return diag;

  // If we remapped the location back to the input, we need to update the
  // components of the diagnostic to account for the new location information.
  auto [newLine, newCol] = sourceMgr.getLineAndColumn(newLoc);
  --newCol;
  int colDiff = diag.getColumnNo() - newCol;

  // Update the diagnostic contents based on the column difference.
  SmallVector<std::pair<unsigned, unsigned>> ranges(diag.getRanges());
  StringRef lineContents = diag.getLineContents();
  if (colDiff) {
    for (auto &range : ranges) {
      range.first -= colDiff;
      range.second -= colDiff;
    }
    if (!lineContents.empty())
      lineContents = lineContents.drop_front(colDiff);
  }

  // Update the locations of the fixits.
  SmallVector<llvm::SMFixIt> fixits;
  for (auto &fixit : diag.getFixIts()) {
    fixits.emplace_back(llvm::SMRange(mapLocation(fixit.getRange().Start),
                                      mapLocation(fixit.getRange().End)),
                        fixit.getText());
  }

  // Remap the file name and record the diagnostic.
  StringRef newFileName =
      sourceMgr.getMemoryBuffer(sourceMgr.FindBufferContainingLoc(newLoc))
          ->getBufferIdentifier();
  return llvm::SMDiagnostic(sourceMgr, newLoc, newFileName, newLine, newCol,
                            diag.getKind(), diag.getMessage(), lineContents,
                            ranges, fixits);
}

//===----------------------------------------------------------------------===//
// Expression Extraction
//===----------------------------------------------------------------------===//

/// Return true if the given line matches any of the given prefixes.
template <typename Prefixes>
static bool matchesAnyPrefix(StringRef line, const Prefixes &prefixes) {
  return llvm::any_of(
      prefixes, [&](StringRef prefix) { return line.starts_with(prefix); });
}

static bool isFunctionOrStructDeclaration(StringRef code) {
  static constexpr auto kPrefixes = {
      "fn ",      "def ",    "struct ", "@adaptive",          "@always_inline",
      "@closure", "@export", "@value",  "@register_passable",
  };
  return matchesAnyPrefix(code, kPrefixes);
}

static bool isIndented(StringRef code) {
  static constexpr auto kPrefixes = {" ", "\t"};
  return matchesAnyPrefix(code, kPrefixes);
}

static bool isSimpleImport(StringRef code) {
  // `import` is a reserved keyword.
  return code.starts_with("import ");
}

static bool isFromImport(StringRef code) {
  // `from` is a reserved keyword.
  return code.starts_with("from ");
}

static bool isAlias(StringRef code) { return code.starts_with("alias "); }

static bool isOpenParenthesis(char c) { return c == '(' || c == '['; }

static bool isCloseParenthesis(char c) { return c == ')' || c == ']'; }

/// Parse the beginning of `unparsedCode` as a simple `import *` statement. If
/// the parsing fails, false is returned. `unparsedCode` is modified to point to
/// the next statement is the parsing was successful, in which case true is
/// returned.
static bool tryHandleSimpleImport(StringRef &unparsedCode,
                                  SmallVectorImpl<StringRef> &topLevelCode) {
  if (!isSimpleImport(unparsedCode))
    return false;
  // It seems that mojo doesn't support simple imports yet.
  auto [line, rest] = unparsedCode.split("\n");
  topLevelCode.push_back(line);
  unparsedCode = rest;
  return true;
}

/// Parse the beginning of `unparsedCode` as a `from * import` statement, a
/// `fn`, a `def` or a `struct` top level statement. If the parsing fails, false
/// is returned. `unparsedCode` is modified to point to the next statement is
/// the parsing was successful, in which case true is returned.
static bool tryHandleFromImportAliasFunctionOrStruct(
    StringRef &unparsedCode, SmallVectorImpl<StringRef> &topLevelCode) {
  bool isFunctionOrStruct = isFunctionOrStructDeclaration(unparsedCode);
  if (!isFunctionOrStruct && !isFromImport(unparsedCode) &&
      !isAlias(unparsedCode))
    return false;

  // These statements can have a hierarchy of () or [], so we need to parse
  // until we have visited all of them.

  // If we are in a function or struct, we also need to find a : outside of any
  // parenthesis.
  bool requiresOuterColon = isFunctionOrStruct;

  // The following block will find the top declaration and not the body of the
  // entity we are parsing. For example, if we have the function
  //
  //   fn foo() -> Int:
  //     return 12
  //
  // then this block find the `fn foo() -> Int:\n`, even if it's split across
  // many lines. The body will be handled later.
  StringRef declStr;
  {
    // This is an iterator of the unparsed code.
    size_t pos = 0;
    // This counts how many unmatched ( or [ we have found so far.
    size_t openings = 0;
    for (size_t end = unparsedCode.size(); pos < end; ++pos) {
      if (unparsedCode[pos] == '\n' && openings == 0 && !requiresOuterColon)
        break;
      // Skip past comments.
      if (unparsedCode[pos] == '#') {
        pos = unparsedCode.find('\n', pos);
        continue;
      }

      if (isOpenParenthesis(unparsedCode[pos]))
        ++openings;
      else if (isCloseParenthesis(unparsedCode[pos]))
        --openings;
      else if (unparsedCode[pos] == ':' && openings == 0)
        requiresOuterColon = false;
    }
    declStr = unparsedCode.substr(0, pos + 1);
    unparsedCode = unparsedCode.substr(pos + 1);
  }

  if (isFunctionOrStruct) {
    // We now absorb all indented code included empty lines, which make the body
    // of the entity we are parsing. This doesn't apply to aliases, for example.
    while (!unparsedCode.empty()) {
      auto [line, rest] = unparsedCode.split("\n");
      if (!line.empty() && !isIndented(line) && !line.ltrim().starts_with("#"))
        break;
      declStr = StringRef(declStr.data(), line.end() - declStr.data());
      unparsedCode = rest;
    }
  }
  topLevelCode.push_back(declStr);
  return true;
}

static void extractExpressionCode(StringRef exprText,
                                  SmallVectorImpl<StringRef> &topLevelCode,
                                  SmallVectorImpl<StringRef> &mainBodyCode) {
  // The following code will consume chunks of code assigning them to either
  // the top-level or the main body sections.
  StringRef unparsedCode = exprText;
  while (!unparsedCode.empty()) {
    // Note: We are not yet handling multiline expressions with \.
    if (!tryHandleFromImportAliasFunctionOrStruct(unparsedCode, topLevelCode) &&
        !tryHandleSimpleImport(unparsedCode, topLevelCode)) {
      // Any other case is just main body code.
      auto [line, rest] = unparsedCode.split("\n");
      if (!line.empty())
        mainBodyCode.push_back(line);
      unparsedCode = rest;
    }
  }
}

//===----------------------------------------------------------------------===//
// Expression Wrapping
//===----------------------------------------------------------------------===//

/// Wrap the provided expression text in a function so that it can be executed.
/// The generated function uses the provided name, and the provided variables
/// are passed via fields to a generated struct that is used as the first
/// argument of the function.
static std::string
wrapExpressionText(MojoParserContext::REPLLocMapper::ExprLocMapper &locMapper,
                   StringRef wrappedFnName, StringRef exprText,
                   ArrayRef<std::pair<StringRef, Type>> variables,
                   bool isFirstREPLCell) {
  // Wrap the expression text in a function so that we can execute it.
  std::string transformedText;
  llvm::raw_string_ostream exprOS(transformedText);

  // Insert a preamble of imports used by the expression wrapper.
  if (isFirstREPLCell) {
    exprOS << "from memory.unsafe import Pointer\n"
           << "from python.python import Python\n"
           << "from python.object import PythonObject\n";
  }

  // Build the input struct, which contains each of the persistent variables.
  exprOS << "struct __mojo_repl_context__:\n";
  for (auto &[name, type] : variables) {
    exprOS << llvm::formatv(
        "  var `{0}`: Pointer[Pointer[__mlir_type.`{1}`]]\n", name, type);
  }
  if (variables.empty())
    exprOS << "  pass\n";
  exprOS << "\n";

  // Extract out the top-level code from the expression code.
  SmallVector<StringRef> topLevelCode, mainBodyCode;
  extractExpressionCode(exprText, topLevelCode, mainBodyCode);

  // Build a mapping for pieces of the input expression and the wrapped
  // expression, enabling seamless location mapping between the two.
  auto emitAndMapCode = [&](StringRef code) {
    locMapper.addMapping(code, exprOS.str().size());
    exprOS << code << "\n";
  };

  // Splat out the top-level code.
  for (StringRef code : topLevelCode)
    emitAndMapCode(code);

  // Generate a wrapper function to handle the extracting function arguments as
  // references.
  exprOS << "fn " << wrappedFnName
         << "(inout __mojo_repl_arg: __mojo_repl_context__):\n"
            "  try:\n"
            "    __mojo_repl_expr_impl__(__mojo_repl_arg";
  for (auto &[name, type] : variables) {
    exprOS << formatv(
        ", __get_address_as_lvalue(__mojo_repl_arg.`{0}`.load().address)",
        name);
  }
  exprOS << ")\n"
            "  except error:\n"
            "    print(\"Error:\", error.value)\n\n";

  // Finally we can generate the actual expression function.
  exprOS << "def __mojo_repl_expr_impl__(inout __mojo_repl_arg: "
            "__mojo_repl_context__";
  for (auto &[name, type] : variables)
    exprOS << llvm::formatv(", inout `{0}`: __mlir_type.`{1}`", name, type);
  exprOS << ") -> None:\n";

  // Splat out the main body code inside of a nested def. This will allow for us
  // to redefine previous variables transparently.
  exprOS << "  @parameter\n"
            "  def __mojo_repl_expr_body__() -> None:\n";

  exprOS << "    var ___lldb_expr_failed = False\n";
  // The following is the other chunk of code just written by the user.
  for (StringRef code : mainBodyCode) {
    exprOS << "    ";
    emitAndMapCode(code);
  }
  exprOS << "    return\n"
            "  __mojo_repl_expr_body__()\n";

  return exprOS.str();
}

//===----------------------------------------------------------------------===//
// Persistent Variables
//===----------------------------------------------------------------------===//

// Simple utility functor for looking up a decl that's known to exist.
static ASTDecl &lookupSingleDecl(ASTDecl &decl, StringRef name) {
  return *decl.lookupInCurrentScope(name).front();
}

/// Process all of the top-level variables defined within the expression body to
/// see which should be persisted. If a variable is persisted, it will be
/// added to the state struct and the expression body will be rewritten to
/// access it via the state struct.
/// TODO: It'd be a bit nicer to have this handled when actually parsing the
/// variables, but for now we do this as a post-processing step.
static void processVariablesForPersistence(MojoParserREPLListener &listener,
                                           ASTDecl &exprFnDecl,
                                           ASTDecl &stateStructDecl) {
  auto exprFn = cast<KGEN::LIT::FuncOp>(exprFnDecl);
  auto stateStruct = cast<KGEN::LIT::StructDeclOp>(stateStructDecl);

  // Grab all of the variables within the expression body and sort them by name,
  // so that we can deterministically process them.
  ASTDecl &exprBodyDecl =
      lookupSingleDecl(exprFnDecl, "__mojo_repl_expr_body__");
  SmallVector<std::pair<StringAttr, ASTDecl *>> variables;
  for (auto &[name, decls] : exprBodyDecl.getDeclsInScope())
    if (decls.size() == 1 && isa<LetRegDeclOp, VarLetDeclOp>(*decls.front()))
      variables.emplace_back(name, decls.front());
  llvm::sort(variables, [](const auto &lhs, const auto &rhs) {
    return lhs.first.getValue() < rhs.first.getValue();
  });

  OpBuilder structBuilder = OpBuilder::atBlockEnd(stateStruct.getBody());
  Value structValue = exprFn.getArgument(0);
  Attribute targetAttr =
      KGEN::ParamOperatorAttr::get(POC::CurrentTarget, /*operands=*/{},
                                   structBuilder.getType<KGEN::TargetType>());

  // Utility functor to check if a variable should be inserted, and if so insert
  // a new field into the persistent state struct. If the variable was
  // persisted, returns a value corresponding to the address of the field.
  // Returns nullptr otherwise.
  auto checkInsertPersistentVar = [&](Operation *varOp, StringAttr name,
                                      PointerType type) {
    mlir::Type elementType = type.getElementAsType();

    // Check if the variable should be persisted.
    if (!listener.shouldPersistVariable(name, elementType))
      return Value();

    // The variable was persisted, insert a new field into the state struct.
    std::string newFieldName = ("__new_repl_var_" + name.strref()).str();
    structBuilder.create<LIT::StructFieldOp>(varOp->getLoc(), newFieldName,
                                             PointerType::get(type),
                                             /*docString=*/DocStringAttr());

    // Materialize a reference to the variable within the function.
    mlir::ImplicitLocOpBuilder builder(varOp->getLoc(), varOp);
    Value fieldGep = builder.create<LIT::StructGEPOp>(
        varOp->getLoc(), PointerType::get(PointerType::get(type)), newFieldName,
        structValue);
    Value fieldLoad = builder.create<POP::LoadOp>(varOp->getLoc(), fieldGep);

    // TODO: Whenever we have globals, we should be able to use a global
    // variable for the address and ensure it gets preserved. For now, we just
    // malloc the memory.
    mlir::Type indexType = structBuilder.getIndexType();
    // Compute the size of the type.
    Attribute sizeOfAttr = KGEN::ParamOperatorAttr::get(
        POC::GetSizeOf,
        {KGEN::ParameterizedTypeConstantAttr::get(elementType),
         cast<TypedAttr>(targetAttr)},
        indexType);
    Value sizeOf = builder.create<KGEN::ParamConstantOp>(
        indexType, cast<TypedAttr>(sizeOfAttr));
    // Compute the alignment of the type.
    Attribute alignOfAttr = KGEN::ParamOperatorAttr::get(
        POC::GetAlignOf,
        {cast<TypedAttr>(KGEN::ParameterizedTypeConstantAttr::get(elementType)),
         cast<TypedAttr>(targetAttr)},
        indexType);
    Value alignOf = builder.create<KGEN::ParamConstantOp>(
        indexType, cast<TypedAttr>(alignOfAttr));
    // Allocate an aligned blob for the variable.
    Value mallocCast = builder.create<POP::AlignedAllocOp>(
        type, ArrayRef<Value>{alignOf, sizeOf});
    builder.create<POP::StoreOp>(mallocCast, fieldLoad);

    // Return a pointer to the new address of the variable.
    return mallocCast;
  };

  for (auto &[name, decl] : variables) {
    // Handle register based let decls. These have an initializer, and never
    // expose the actual pointer.
    if (auto letOp = dyn_cast<LIT::LetRegDeclOp>(*decl)) {
      Value field = checkInsertPersistentVar(letOp, letOp.getNameAttr(),
                                             PointerType::get(letOp.getType()));
      if (!field)
        continue;
      decl->setIRValue(MRValue(field));

      // Store the value in the persistent state struct.
      OpBuilder builder(letOp);
      builder.create<POP::StoreOp>(letOp.getLoc(), letOp.getValue(), field);

      // Replace all references of the original decl with the initializer.
      letOp.replaceAllUsesWith(letOp.getValue());
      letOp.erase();
      continue;
    }
    // Handle memory based let decls.
    if (auto letOp = dyn_cast<LIT::VarLetDeclOp>(*decl)) {
      if (Value field = checkInsertPersistentVar(letOp, letOp.getNameAttr(),
                                                 letOp.getType())) {
        decl->setIRValue(MRValue(field));
        letOp.replaceAllUsesWith(field);
        letOp.erase();
      }
      continue;
    }
  }
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a diagnostic handler for REPL cells.
class REPLDiagnosticHandler {
public:
  REPLDiagnosticHandler(MojoParserREPLListener &listener,
                        MojoParserContext::REPLLocMapper &locMapper,
                        StringRef exprText, llvm::SourceMgr &sourceMgr)
      : listener(listener), locMapper(locMapper), exprText(exprText) {
    sourceMgr.setDiagHandler(handleDiagnostic, this);
  }

  /// This method processes all of the diagnostics that have been collected.
  /// `exprText` is the text of the wrapped expression that was parsed.
  LogicalResult processDiagnostics();

private:
  /// A static diagnostic handler function that is usable with SourceMgr. This
  /// handler simply collects diagnostics, which will get processed later.
  static void handleDiagnostic(const llvm::SMDiagnostic &diagnostic,
                               void *ctx) {
    auto *handler = static_cast<REPLDiagnosticHandler *>(ctx);
    handler->diagnostics.emplace_back(
        handler->locMapper.mapDiagnostic(diagnostic));
  }

  MojoParserREPLListener &listener;
  MojoParserContext::REPLLocMapper &locMapper;
  StringRef exprText;
  std::vector<llvm::SMDiagnostic> diagnostics;
};
} // namespace

LogicalResult REPLDiagnosticHandler::processDiagnostics() {
  if (diagnostics.empty())
    return success();

  // Notify the listener of the diagnostics.
  listener.notifyDiagnostics(diagnostics);

  // Process all of the diagnostics to check for errors, and apply fixits if
  // possible.

  // This takes advantage of the fact that fixits are ordered to apply multiple
  // fixits to a single expression.
  std::string newText;
  size_t prevEnd = 0;
  auto applyFixit = [&](const llvm::SMFixIt &fixit) -> LogicalResult {
    llvm::SMRange range = fixit.getRange();
    if (!range.isValid())
      return failure();

    StringRef removedText(range.Start.getPointer(),
                          range.End.getPointer() - range.Start.getPointer());
    StringRef insertedText = fixit.getText();

    // The current range starts at the previous end pointer.
    StringRef currentOriginalRange(exprText.begin() + prevEnd);

    // Add the substring from the start of the current original text range.
    if (range.Start.getPointer() < currentOriginalRange.end() &&
        range.Start.getPointer() >= currentOriginalRange.begin())
      newText += currentOriginalRange.substr(
          0, range.Start.getPointer() - currentOriginalRange.begin());

    // Add the text to insert.
    newText += insertedText;

    // Update prevEnd. At the *very* end, we will clean up by adding the
    // remaining substring. Subtract off the size of the inserted text because
    // the pointers are all indexed off the original text.
    prevEnd += range.End.getPointer() - currentOriginalRange.begin();
    return success();
  };

  bool hadFixit = false;
  bool allDiagsHandled = llvm::all_of(diagnostics, [&](const auto &diag) {
    if (diag.getFixIts().empty())
      return diag.getKind() != llvm::SourceMgr::DK_Error;

    hadFixit = true;
    return llvm::all_of(diag.getFixIts(), [&](const llvm::SMFixIt &fixit) {
      return succeeded(applyFixit(fixit));
    });
  });

  // If we handled all the diagnostics and we applied fixits, notify the
  // listener that we have an improved expression.
  if (allDiagsHandled && hadFixit) {
    // Complete fixit handling by adding the substring from prevEnd to the end
    // of the buffer. We do this here because we only want to do it if/once
    // *all* diagnostics are handled.
    newText += exprText.substr(prevEnd);
    listener.notifyFixedExpr(newText);
  }

  return success(allDiagsHandled);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Build a module decl for use in a REPL expression.
static ASTDecl &buildREPLModule(const llvm::MemoryBuffer *sourceBuf,
                                SharedState &sharedState) {
  StringRef exprId = sourceBuf->getBufferIdentifier();

  // If we are emitting debug info, create a file entry for this file.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (sharedState.diBuilder)
    fileGuard = sharedState.diBuilder->pushFile(exprId, "/");

  // Create the input module.
  MLIRContext *ctx = sharedState.getContext();
  auto fileLoc = FileLineColLoc::get(ctx, exprId, /*line=*/0, /*column=*/0);
  return sharedState.createModule(exprId, sourceBuf, fileLoc);
}

/// Build and resolve a REPL module for the given wrapped expression string.
/// Returns the fully resolved REPL module decl.
static ASTDecl &
buildAndResolveREPLModule(const llvm::MemoryBuffer *sourceBuf,
                          SharedState &sharedState,
                          ArrayRef<KGEN::LIT::ASTDecl *> replModuleDecls) {
  ASTDecl &moduleDecl = buildREPLModule(sourceBuf, sharedState);

  // Before resolving everything in the REPL cell, resolve the body and import
  // as many of the previously defined REPL decls that we can.
  if (!replModuleDecls.empty()) {
    ASTDecl *lastModuleDecl = replModuleDecls.back();

    // Explicitly import any decls from the previous REPL module that aren't
    // already defined in the current module. We can't use wildcards here
    // because we also want to import _ and other traditionally "hidden" decls
    // from previous cells.
    SmallVector<std::pair<StringAttr, const TinyPtrVector<ASTDecl *>>> fnDecls;
    auto &moduleChildDecls = moduleDecl.getDeclsInScope();
    for (auto &[name, decls] : lastModuleDecl->getDeclsInScope()) {
      auto existingDeclsIt = moduleChildDecls.find(name);
      if (existingDeclsIt == moduleChildDecls.end()) {
        sharedState.declResolver->aliasDecls(decls, name, SMLoc(), moduleDecl);
        continue;
      }
      // If we hit an overlap and these are function decls, save them for
      // processing for later. We might be able to import if the signatures
      // don't overlap.
      if (isa<LIT::FuncOp>(*existingDeclsIt->second.front()) &&
          isa<LIT::FuncOp>(*decls.front())) {
        fnDecls.push_back({name, decls});
      }
    }

    // Now that we've imported all of the decls we can, go ahead and import the
    // functions that have name overlaps. We do this afterwards so that we can
    // resolve the signature of the pre-existing functions to see if there are
    // signature overlaps (to avoid duplicate function declarations).
    for (auto &[name, decls] : fnDecls) {
      (void)sharedState.declResolver->tryAliasDecls(decls, name, SMLoc(),
                                                    moduleDecl);
    }
  }

  // With the top-level of the file parsed, we can now go ahead and resolve all
  // of the deferred declarations.
  sharedState.declResolver->resolveAll();
  return moduleDecl;
}

MojoParserContext::REPLLocMapper &MojoParserContext::getREPLLocMapper() {
  return impl->replLocMapper;
}

MojoASTDeclRef MojoParserContext::parseREPLExpresion(
    MojoParserREPLListener &listener, unsigned exprFileId,
    StringRef replExprFnName,
    ArrayRef<std::pair<StringRef, Type>> replVariables) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  const llvm::MemoryBuffer *exprFileBuf = sourceMgr.getMemoryBuffer(exprFileId);
  StringRef exprText = exprFileBuf->getBuffer();

  // Build a location mapper for this expression.
  impl->replLocMapper.exprMappers.emplace_back(
      std::make_unique<REPLLocMapper::ExprLocMapper>(exprText));
  REPLLocMapper::ExprLocMapper &exprLocMapper =
      *impl->replLocMapper.exprMappers.back();

  // Set up a diagnostic handler to process diagnostics emitted during parsing.
  auto oldDiagHandler = sourceMgr.getDiagHandler();
  auto oldDiagContext = sourceMgr.getDiagContext();
  auto resetHandlerOnExit = llvm::make_scope_exit(
      [&] { sourceMgr.setDiagHandler(oldDiagHandler, oldDiagContext); });
  REPLDiagnosticHandler diagHandler(listener, impl->replLocMapper, exprText,
                                    sourceMgr);

  // Wrap the expression text in a function so that we can execute it.
  std::string wrappedExprText =
      wrapExpressionText(exprLocMapper, replExprFnName, exprText, replVariables,
                         /*isFirstREPLCell=*/impl->replModuleDecls.empty());
  listener.notifyWrappedExpr(wrappedExprText);

  // TODO: We should print the expression to a file if we need debug
  // information attached.
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
      wrappedExprText, ("wrapped " + exprFileBuf->getBufferIdentifier()).str());
  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(
      sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc()));
  exprLocMapper.setWrappedExpr(sourceBuf->getBuffer());

  // Resolve a module decl for this REPL expression.
  ASTDecl &moduleDecl = buildAndResolveREPLModule(sourceBuf, impl->sharedState,
                                                  impl->replModuleDecls);

  // Clear up the error state so that we are still able to parse future cells,
  // we'll handle diagnostic checks below.
  impl->sharedState.diags.clear();

  // Check if we have a non-recoverable parse error, or emitted an error and
  // then recovered.
  if (failed(diagHandler.processDiagnostics()) ||
      failed(mlir::verify(*impl->module))) {
    // In the case of failure, remove the module so that it doesn't prevent
    // parsing future cells.
    impl->detachedREPLModules.push_back(moduleDecl.getIfOperation());
    moduleDecl.getIfOperation()->remove();
    return nullptr;
  }

  // Process variables within the expression function for persistence.
  processVariablesForPersistence(
      listener, lookupSingleDecl(moduleDecl, "__mojo_repl_expr_impl__"),
      lookupSingleDecl(moduleDecl, "__mojo_repl_context__"));

  // Update the last REPL module decl.
  impl->replModuleDecls.push_back(&moduleDecl);
  return MojoASTDeclRef(&lookupSingleDecl(moduleDecl, replExprFnName));
}

std::vector<Mojo::CodeCompletionResult>
MojoParserContext::codeCompleteREPLExpresion(
    StringRef exprText, uint64_t completionPosition,
    ArrayRef<std::pair<StringRef, Type>> replVariables) {
  // Insert a marker into the expression text at the completion position. This
  // is only really necessary because we currently do string splicing to fake
  // support for top-level code, meaning that we don't know where the completion
  // position will end up after that process is done.
  constexpr StringLiteral kCompletionMarker = "<#COMPLETION_MARKER#>";
  std::string exprTextWithMarker = exprText.substr(0, completionPosition).str();
  exprTextWithMarker += kCompletionMarker;
  exprTextWithMarker += exprText.drop_front(completionPosition).str();

  // Build a location mapper for this expression.
  REPLLocMapper locMapper(getSourceMgr());
  locMapper.exprMappers.emplace_back(
      std::make_unique<REPLLocMapper::ExprLocMapper>(exprText));

  // Wrap the expression text in a function so that we can execute it.
  std::string wrappedExprText = wrapExpressionText(
      *locMapper.exprMappers.back(), "__repl_code_complete_fn",
      exprTextWithMarker, replVariables,
      /*isFirstREPLCell=*/impl->replModuleDecls.empty());

  // Remove the completion marker from the wrapped expression text and grab the
  // new completion position.
  completionPosition = wrappedExprText.find(kCompletionMarker);
  wrappedExprText.erase(completionPosition, kCompletionMarker.size());

  // Functor used to parse a REPL expression for use by code completion.
  auto replParseFn = [&](MojoParserContext &ctx, int fileId) {
    SourceMgr &mainSourceMgr = impl->sharedState.getSourceMgr();
    SourceMgr &sourceMgr = ctx.getSourceMgr();
    const llvm::MemoryBuffer *sourceBuf =
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

    // Pull in the existing REPL module state.
    SmallVector<ASTDecl *> completionReplModuleDecls;
    for (ASTDecl *module : impl->replModuleDecls) {
      int bufferId = mainSourceMgr.FindBufferContainingLoc(module->getLoc());
      const llvm::MemoryBuffer *moduleBuf =
          mainSourceMgr.getMemoryBuffer(bufferId);

      // Add the copy of the decl and resolve its body.
      int completionBufferId = sourceMgr.AddNewSourceBuffer(
          llvm::MemoryBuffer::getMemBuffer(*moduleBuf),
          SMLoc::getFromPointer(sourceBuf->getBufferStart()));
      ASTDecl &newModuleDecl = buildAndResolveREPLModule(
          sourceMgr.getMemoryBuffer(completionBufferId), ctx.impl->sharedState,
          completionReplModuleDecls);
      completionReplModuleDecls.push_back(&newModuleDecl);
    }

    // Resolve a module decl for this REPL expression.
    buildAndResolveREPLModule(sourceBuf, ctx.impl->sharedState,
                              completionReplModuleDecls);
  };

  return MojoParserContext::codeComplete(
      llvm::MemoryBufferRef(wrappedExprText, ""), completionPosition,
      impl->sharedState.getContext(), impl->sharedState.runtime,
      impl->sharedState.options, replParseFn);
}

void MojoParserContext::removeLastREPLExpression() {
  assert(!impl->replModuleDecls.empty() && "expected at least one REPL module");
  ASTDecl *moduleDecl = impl->replModuleDecls.pop_back_val();
  impl->detachedREPLModules.push_back(moduleDecl->getIfOperation());
  moduleDecl->getIfOperation()->remove();
}
