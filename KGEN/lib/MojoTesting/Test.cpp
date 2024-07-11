//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTesting/Test.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/Paths.h"
#include "Support/Process.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include <filesystem>

#ifndef _WIN32_
#include <unistd.h>
#endif

using namespace M;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;
using namespace mlir::lsp;

//===----------------------------------------------------------------------===//
// TestID
//===----------------------------------------------------------------------===//

TestID::TestID(const Twine &argID) : id(argID.str()) {
  // Parse the test ID into its individual components.
  StringRef pathAndSuite;
  std::tie(pathAndSuite, test) = strref().split("::");
  std::tie(path, testSuite) = pathAndSuite.split('@');
}
TestID::TestID(StringRef pathArg, StringRef testSuiteArg, StringRef testArg) {
  llvm::raw_string_ostream(id)
      << pathArg << (testSuiteArg.empty() ? "" : ("@" + testSuiteArg))
      << (testArg.empty() ? "" : ("::" + testArg));

  path = strref().take_front(pathArg.size());
  if (!testSuiteArg.empty())
    testSuite = strref().substr(pathArg.size() + 1, testSuiteArg.size());
  if (!testArg.empty())
    test = strref().take_back(testArg.size());
}

TestID &TestID::operator=(const TestID &rhs) {
  id = rhs.id;
  path = rhs.path;
  testSuite = rhs.testSuite;
  test = rhs.test;
  return *this;
}

std::filesystem::path TestID::getFilePath() const {
  std::error_code ec;
  auto resultPath = std::filesystem::canonical(path, ec);
  if (ec)
    return path;
  return resultPath;
}

TestID TestID::withTest(StringRef test) const {
  return TestID(path, testSuite, test);
}

ErrorOr<SmallVector<std::string>>
TestID::parseScopedName(StringRef scopedName) {
  SmallVector<std::string> result;
  while (!scopedName.empty()) {
    StringRef scope = scopedName;
    if (scopedName.contains(".")) {
      std::tie(scope, scopedName) = scopedName.split('.');
      if (scope.empty() || scopedName.empty()) {
        StringRef expectedLoc = scope.empty() ? "before" : "after";
        return Error("empty scope in test name, expected name " + expectedLoc +
                     " '.'");
      }
    } else {
      scopedName = {};
    }

    // If there are no escaped characters, we can use the name directly.
    if (!scope.contains("\\")) {
      result.emplace_back(scope.str());
      continue;
    }

    // Otherwise, pull in characters and check for escaped characters.
    std::string &unescaped = result.emplace_back();
    for (unsigned i = 0, e = scope.size(); i != e; ++i) {
      if (scope[i] == '\\') {
        if ((i + 2) >= e)
          return Error("invalid escape sequence in test name");
        unsigned char c = llvm::hexDigitValue(scope[++i]);
        unescaped.push_back((c << 4) | llvm::hexDigitValue(scope[++i]));
      } else {
        unescaped.push_back(scope[i]);
      }
    }
  }
  if (result.empty())
    return Error("empty test name");
  return std::move(result);
}

raw_ostream &KGEN::Mojo::operator<<(raw_ostream &os, const TestID &testID) {
  return os << testID.strref();
}

//===----------------------------------------------------------------------===//
// JSON Serialization

bool KGEN::Mojo::fromJSON(const llvm::json::Value &value, TestID &result,
                          llvm::json::Path path) {
  if (std::optional<StringRef> resultId = value.getAsString()) {
    result = TestID(resultId->str());
    return true;
  }
  return false;
}

llvm::json::Value KGEN::Mojo::toJSON(const TestID &value) {
  return llvm::json::Value(value.id);
}

//===----------------------------------------------------------------------===//
// TestExecutionResult
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Display

void TestExecutionResult::print(raw_ostream &os) const {
  // Walk the test tree collecting failures.
  size_t numPassed = 0, numSkipped = 0;
  SmallVector<const TestExecutionResult *, 8> failures;
  SmallVector<const TestExecutionResult *> worklist(1, this);
  while (!worklist.empty()) {
    const TestExecutionResult *result = worklist.pop_back_val();

    // If the result has children, don't count this result, just process the
    // children.
    if (!result->children.empty()) {
      for (const TestExecutionResult &child : result->children)
        worklist.push_back(&child);
      continue;
    }

    switch (result->kind) {
    case TestExecutionResult::kSuccess:
      ++numPassed;
      break;
    case TestExecutionResult::kSkipped:
      ++numSkipped;
      break;
    default:
      failures.push_back(result);
      break;
    }
  }

  // Print the results.
  os << llvm::formatv("Testing Time: {0}.{1:02}s\n\n", duration.count() / 1000,
                      duration.count() % 1000);

  size_t totalTests = numPassed + numSkipped + failures.size();
  os << "Total Discovered Tests: " << totalTests << "\n\n";

  // Print the high level results of execution:
  auto printResult = [&](StringRef label, size_t count) {
    os << label << count
       << llvm::formatv(" ({0:p})\n", (count * 1.0) / totalTests);
  };
  printResult("Passed : ", numPassed);
  printResult("Failed : ", failures.size());
  printResult("Skipped: ", numSkipped);

  // Print more complete information for the failures.
  std::string marker(20, '*');
  for (const TestExecutionResult *result : failures) {
    os << llvm::formatv("\n{0} Failure: '{1}' {0}\n", marker, result->testID);
    if (!result->error.empty())
      os << "\n" << result->error << "\n";
    if (!result->stdOut.empty())
      os << "\n" << result->stdOut << "\n";
    if (!result->stdErr.empty())
      os << "\n" << result->stdErr << "\n";
    os << marker << "\n";
  }
}

//===----------------------------------------------------------------------===//
// JSON Serialization

/// Stringify a test execution result kind.
static std::string stringifyResultKind(TestExecutionResult::Kind kind) {
  switch (kind) {
  case TestExecutionResult::kSuccess:
    return "success";
  case TestExecutionResult::kInitializationError:
    return "initializationError";
  case TestExecutionResult::kExecutionError:
    return "executionError";
  case TestExecutionResult::kSkipped:
    return "skipped";
  }
  llvm_unreachable("unknown test execution result kind");
}

/// Parse a test execution result kind from a string.
static std::optional<TestExecutionResult::Kind>
parseResultKind(StringRef kind) {
  return llvm::StringSwitch<std::optional<TestExecutionResult::Kind>>(kind)
      .Case("success", TestExecutionResult::kSuccess)
      .Case("initializationError", TestExecutionResult::kInitializationError)
      .Case("executionError", TestExecutionResult::kExecutionError)
      .Case("skipped", TestExecutionResult::kSkipped)
      .Default(std::nullopt);
}

bool KGEN::Mojo::fromJSON(const llvm::json::Value &value,
                          TestExecutionResult &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;

  // Parse the kind of the result.
  std::string kindStr;
  if (!o.map("kind", kindStr))
    return false;
  std::optional<TestExecutionResult::Kind> kind = parseResultKind(kindStr);
  if (!kind)
    return false;
  result.kind = *kind;

  // Parse the duration of the result.
  int64_t duration;
  if (!o.map("duration_ms", duration))
    return false;
  result.duration = std::chrono::milliseconds(duration);

  return o.mapOptional("children", result.children) &&
         o.map("testID", result.testID) && o.map("error", result.error) &&
         o.map("stdErr", result.stdErr) && o.map("stdOut", result.stdOut);
}

llvm::json::Value KGEN::Mojo::toJSON(const TestExecutionResult &value) {
  llvm::json::Object object{
      {"kind", stringifyResultKind(value.kind)},
      {"duration_ms", value.duration.count()},
      {"testID", value.testID},
      {"error", value.error},
      {"stdErr", value.stdErr},
      {"stdOut", value.stdOut},
  };
  if (!value.children.empty())
    object["children"] = llvm::json::Value(value.children);
  return object;
}

//===----------------------------------------------------------------------===//
// Test
//===----------------------------------------------------------------------===//

void Test::print(raw_ostream &os) const {
  os << "<" << testID << ">";
  if (children.empty())
    return;
  os << "\n";

  mlir::raw_indented_ostream indentedOS(os);
  llvm::interleave(children, indentedOS.indent(2), "\n");
}

raw_ostream &KGEN::Mojo::operator<<(raw_ostream &os, const Test &test) {
  test.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// Test: Discovery

/// Return if the given operation defines a test suite.
static bool definesTestSuite(Operation *op) {
  return llvm::isa_and_present<FileModuleOp, PackageOp>(op);
}

/// A type corresponding to an async test discovery result.
using AsyncOptionalTest = LLCL::AsyncValueRef<std::optional<Test>>;

/// Await the result of an async result, returning an easier to manipulate
/// form.
static ErrorOr<std::optional<Test>> awaitTest(AnyAsyncValueRef result) {
  await(result);
  if (result.isError())
    return std::move(result.takeDiagnostic().getMessage());
  return std::move(result.get<std::optional<Test>>());
}

/// This class is used to discover tests from within a specific Mojo file.
struct Test::TestDiscovery {
  TestDiscovery(LLCL::Runtime &runtime,
                ArrayRef<std::string> additionalImportPaths)
      : runtime(runtime), additionalImportPaths(additionalImportPaths) {
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }

  //===--------------------------------------------------------------------===//
  // Utilities

  /// Return a TestID used for referencing the given decl.
  static TestID getDeclTestID(StringRef path, Operation *op,
                              bool isDocTestSuite = false) {
    auto printSymbol = [](StringRef symbol) {
      // Print the symbol name, escaping any characters that are not printable
      // or conflict with the ID schema.
      std::string result;
      llvm::raw_string_ostream os(result);
      for (unsigned char c : symbol) {
        if (llvm::isPrint(c) && !llvm::is_contained(". \"'\\`:", c))
          os << c;
        else
          os << '\\' << llvm::hexdigit(c >> 4) << llvm::hexdigit(c & 0x0F);
      }
      return result;
    };
    mlir::SymbolOpInterface symOp = cast<mlir::SymbolOpInterface>(op);

    // Collect the symbols making up the test name if we're not defining a
    // test suite.
    SmallVector<std::string> symbols;
    if (!isDocTestSuite) {
      do {
        if (definesTestSuite(symOp))
          break;
        symbols.push_back(printSymbol(symOp.getNameAttr()));
      } while ((symOp = symOp->getParentOfType<mlir::SymbolOpInterface>()));
    }
    std::string testName = llvm::join(llvm::reverse(symbols), ".");

    // Collect the symbols making up the test suite name.
    symbols.clear();
    if (isDocTestSuite)
      symbols.push_back("__doc__");
    do {
      // The top-level id is already encoded in the path, no need to add it
      // to the test suite.
      if (isa<ModuleOp, PackageOp>(symOp->getParentOp()))
        break;
      symbols.push_back(printSymbol(symOp.getNameAttr()));
    } while ((symOp = symOp->getParentOfType<mlir::SymbolOpInterface>()));
    std::string testSuiteName = llvm::join(llvm::reverse(symbols), ".");

    // If this is a package, use the filename location of the symbol for the
    // path. This ensures we grab the right source path for the test id.
    if (isa<PackageOp>(symOp->getParentOp()))
      path = symOp->getLoc()->findInstanceOf<FileLineColLoc>().getFilename();

    return TestID(path, testSuiteName, testName);
  }
  static TestID getDeclTestID(StringRef path, MojoASTDeclRef decl) {
    return getDeclTestID(path, decl->getIfOperation());
  }

  /// Return if the given decl defines a unit test.
  static bool doesDeclDefineUnitTest(Operation *declOp, ASTDecl &context,
                                     SharedState &shared) {
    // Tests are defined by functions at starting with `test_` or ends with
    // `_test`, that are defined at the top-level scope.
    auto fn = dyn_cast_if_present<FuncOp>(declOp);
    if (!fn || !isa<FileModuleOp>(declOp->getParentOp()) ||
        !isa<ModuleOp>(fn->getParentOp()->getParentOp()))
      return false;

    std::optional<StringRef> name = fn.getSourceName();
    if (!name || !(name->starts_with("test_") || name->ends_with("_test")))
      return false;
    auto fnSignature = fn.getSignature();

    // Validate that the test has the expected signature.
    if (!fnSignature.getParamTypes().empty() ||
        !fnSignature.getResultParamTypes().empty())
      return false;
    ASTType resultType(fn.getUserResultType());
    ArrayRef<Type> argTypes = fnSignature.getArguments();

    // Check for a test of the form: `fn test()` or `fn test() raises`.
    if (resultType.isNoneType()) {
      return argTypes.empty() ||
             (fnSignature.isThrows() && argTypes.size() == 2);
    }

    // Otherwise, check for a test of the form `def test()`. This form returns
    // an object and raises.
    return resultType.isEqualCanon(
               shared.lookupObjectType(context, context.getLoc())) &&
           fnSignature.isThrows() && argTypes.size() == 2;
  }

  static bool doesDeclDefineUnitTest(MojoASTDeclRef decl, SharedState &shared) {
    return doesDeclDefineUnitTest(decl->getIfOperation(), *decl, shared);
  }

  /// Return the source manager buffer id containing the given operation.
  static std::optional<int> getBufferIDForOp(Operation *op,
                                             SourceMgr &sourceMgr) {
    if (auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>()) {
      for (int i = 1, e = sourceMgr.getNumBuffers(); i <= e; ++i) {
        const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(i);
        if (buffer->getBufferIdentifier() == loc.getFilename())
          return i;
      }
    }
    return std::nullopt;
  }

  /// Return a doc test suite defined by the given decl, or nullopt if no doc
  /// tests are defined.
  static std::optional<Test> getDocTestSuiteFromDecl(StringRef filePath,
                                                     Operation *declOp,
                                                     SourceMgr &sourceMgr,
                                                     int bufId = 0) {
    auto astDeclOp = dyn_cast_if_present<ASTDeclInterface>(declOp);
    if (!astDeclOp)
      return std::nullopt;
    DocStringAttr docStringAttr = astDeclOp.getDocStringAttr();
    if (!docStringAttr)
      return std::nullopt;
    DocString docString(docStringAttr);
    auto codeBlocks = docString.getCodeBlocks();
    if (codeBlocks.empty())
      return std::nullopt;
    TestID testID(getDeclTestID(filePath, declOp, /*isDocTestSuite=*/true));

    // Grab the location of the doc string in the source buffer.
    SMLoc docStartLoc;
    if (FileLineColLoc docLocAttr = docString.getLoc()) {
      // If we don't have a buffer, try to find it from the operation location.
      if (!bufId)
        bufId = getBufferIDForOp(declOp, sourceMgr).value_or(0);
      if (bufId) {
        docStartLoc = sourceMgr.FindLocForLineAndColumn(
            bufId, docLocAttr.getLine(), docLocAttr.getColumn());
      }
    }
    StringRef rawDocStr = docStringAttr.getString();

    std::vector<Test> children;
    for (auto [index, block] : llvm::enumerate(docString.getCodeBlocks())) {
      // Compute the location for the code block. This involves mapping the
      // code block location to the location in the decl buffer.
      std::optional<Test::SourceRange> location;
      if (docStartLoc.isValid()) {
        StringRef code = block.getRawCode();
        SMLoc blockLoc = SMLoc::getFromPointer(
            docStartLoc.getPointer() + (code.data() - rawDocStr.data()));
        SMLoc blockEndLoc =
            SMLoc::getFromPointer(blockLoc.getPointer() + code.size() + 1);
        auto [line, col] = sourceMgr.getLineAndColumn(blockLoc, bufId);
        auto [endLine, endCol] = sourceMgr.getLineAndColumn(blockEndLoc, bufId);
        if (line && col && endLine && endCol)
          location.emplace(line - 1, col, endLine, endCol);
      }
      children.emplace_back(Test(testID.withTest(Twine(index).str()),
                                 /*newChildren=*/{}, location));
    }
    return Test(TestID(testID), std::move(children));
  }
  static std::optional<Test> getDocTestSuiteFromDecl(StringRef filePath,
                                                     MojoASTDeclRef decl,
                                                     SourceMgr &sourceMgr) {
    if (Operation *op = decl->getIfOperation()) {
      return getDocTestSuiteFromDecl(
          filePath, op, sourceMgr,
          sourceMgr.FindBufferContainingLoc(decl->getLoc()));
    }
    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // Discover: Mojo Source

  /// Return a source range for the given decl.
  static std::optional<Test::SourceRange>
  getSourceRangeForDecl(MojoASTDeclRef decl, SourceMgr &sourceMgr) {
    Operation *op = decl->getIfOperation();
    if (!op)
      return std::nullopt;
    if (auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>()) {
      int line = loc.getLine(), col = loc.getColumn();
      return Test::SourceRange{line, col, line, col};
    }
    return std::nullopt;
  }

  /// Discover tests defined by given Mojo decl.
  static void discoverTestsInDecl(StringRef path, MojoASTDeclRef ref,
                                  SharedState &shared, std::vector<Test> &tests,
                                  bool processUnitTests) {
    // Check if the decl defines a unit test.
    if (processUnitTests && doesDeclDefineUnitTest(ref, shared)) {
      tests.emplace_back(
          Test(getDeclTestID(path, ref), /*newChildren=*/{},
               getSourceRangeForDecl(ref, shared.getSourceMgr())));
    }

    // Check for a doc test suite if this isn't a package (package doc strings
    // are just a copy from their __init__).
    if (!isa<PackageOp>(*ref)) {
      if (std::optional<Test> test =
              getDocTestSuiteFromDecl(path, ref, shared.getSourceMgr()))
        tests.emplace_back(std::move(*test));
    }
  }

  /// Discover tests defined within the given Mojo decl.
  static void discoverTestsNestedInDecl(StringRef path, MojoASTDeclRef ref,
                                        SharedState &shared,
                                        std::vector<Test> &tests,
                                        bool processUnitTests) {
    // Process tests in the decl.
    discoverTestsInDecl(path, ref, shared, tests, processUnitTests);

    // Process tests in the children of the decl.
    for (const MojoASTDeclRef::ChildEntry &child : ref.getChildren()) {
      for (MojoASTDeclRef decl : child.getDecls()) {
        // We only process direct children.
        if (decl->getParentDecl() != &*ref)
          continue;

        // If the decl doesn't define a test suite, immediately process tests it
        // defines.
        if (!definesTestSuite(decl.getIfOperation())) {
          // If the operation is still a container, recurse into it but don't
          // start a new test suite.
          if (isa<StructDeclOp, TraitDeclOp>(*decl)) {
            discoverTestsNestedInDecl(path, decl, shared, tests,
                                      processUnitTests);
          } else {
            discoverTestsInDecl(path, decl, shared, tests, processUnitTests);
          }
          continue;
        }

        // Otherwise, collect all tests in the decl as a new test suite.
        std::vector<Test> children;
        discoverTestsNestedInDecl(path, decl, shared, children,
                                  processUnitTests);
        if (!children.empty()) {
          tests.emplace_back(
              Test(getDeclTestID(path, decl), std::move(children)));
        }
      }
    }
  }

  AsyncOptionalTest discoverTestsInMojoSource(const std::filesystem::path &path,
                                              StringRef suiteName = {}) {
    auto asyncResult = AsyncOptionalTest::allocate(runtime);
    LLCL::addTask(runtime, [this, path, suiteName,
                            asyncResult = asyncResult.copy()]() mutable {
      ErrorOr<std::optional<Test>> result =
          discoverTestsInMojoSourceSync(path, suiteName);
      if (result.isError()) {
        return std::move(asyncResult)
            .setToError(EncodedDiagnostic(
                result.takeError(),
                LLCL::UnknownLocationDecoder::getEncodedLocation()));
      }
      std::move(asyncResult).emplace(std::move(*result));
    });
    return asyncResult;
  }

  ErrorOr<std::optional<Test>>
  discoverTestsInMojoSourceSync(const std::filesystem::path &path,
                                StringRef suiteName = {}) {
    KGEN::CompilationOptions compilationOptions;
    ParserConfig parserConfig(&ctx, compilationOptions);

    // Process the mojo file, ignoring any diagnostics emitted along the way
    // (we don't care about emitting errors here, just discovering tests).
    llvm::SourceMgr sourceMgr;
    sourceMgr.setIncludeDirs(additionalImportPaths);
    sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &diag, void *) {});
    MojoParserContext parserContext(sourceMgr, parserConfig);
    MojoASTDeclRef moduleDecl = parserContext.parseFileOrPackage(path);
    if (!moduleDecl || !moduleDecl.getIfOperation())
      return std::nullopt;

    // Process the case where we're looking for a specific test suite.
    MojoASTDeclRef decl = moduleDecl;
    if (!suiteName.empty()) {
      // If this is a doc test suite for the top-level decl, handle that
      // immediately.
      if (suiteName == "__doc__")
        return getDocTestSuiteFromDecl(path.string(), decl, sourceMgr);

      // Parse the scoped name into a list of decl names.
      bool isDocTestSuite = suiteName.consume_back(".__doc__");
      ErrorOr<SmallVector<std::string>> scopes =
          TestID::parseScopedName(suiteName);
      if (scopes.isError())
        return scopes.takeError();

      // If we're looking for a doc test suite, these can be very deep in a
      // symbol stack, and kind of difficult to resolve via ASTDecl scopes.
      // For these, use the symbol table for resolution.
      if (isDocTestSuite) {
        Operation *op = moduleDecl->getIfOperation();
        mlir::SymbolTableCollection symbolTable;
        for (StringRef it : *scopes)
          if (!(op = symbolTable.lookupSymbolIn(op, StringAttr::get(&ctx, it))))
            return std::nullopt;
        return getDocTestSuiteFromDecl(path.string(), op, sourceMgr);
      }

      // Otherwise, this is a normal test suite, for these we can resolve the
      // decl directly from the ASTDecl scopes.
      for (StringRef scope : *scopes) {
        auto decls = decl->lookupInCurrentScope(scope);
        if (decls.size() != 1)
          return std::nullopt;
        decl = decls.front();
      }
    }
    // If the decl doesn't define a test suite, we're done.
    if (!definesTestSuite(decl.getIfOperation()))
      return std::nullopt;
    // We only process unit tests for certain paths.
    std::string pathStem = path.stem().string();
    bool processUnitTests = StringRef(pathStem).starts_with("test_") ||
                            StringRef(pathStem).ends_with("_test");

    // Process the decl to discover tests.
    std::vector<Test> tests;
    discoverTestsNestedInDecl(path.string(), decl,
                              parserContext.getSharedState(), tests,
                              processUnitTests);
    if (tests.empty())
      return std::nullopt;
    return Test(TestID(path.string(), suiteName), std::move(tests));
  }

  //===--------------------------------------------------------------------===//
  // Discover: FileSystem

  /// Discovers tests defined within in the given directory. If `ignoreModules`
  /// is true, nested source modules and packages are ignored for collection.
  AnyAsyncValueRef discoverTestsInDirectory(const std::filesystem::path &path,
                                            bool ignoreModules = false) {
    std::vector<AnyAsyncValueRef> asyncChildren;

    // If the path is a mojo source package, we parse can directly parse out
    // the tests from the package.
    bool hasSourcePackageChild = false;
    if (Filesystem::isMojoSourcePackagePath(path)) {
      // Process this package if we're allowed. Any nested packages will be
      // handled as part of this package, so we can ignore them moving forward.
      if (!ignoreModules) {
        asyncChildren.emplace_back(discoverTestsInMojoSource(path));
        hasSourcePackageChild = ignoreModules = true;
      }
    } else {
      // If this wasn't a package directory, we don't need to ignore nested
      // modules.
      ignoreModules = false;
    }

    // Otherwise, recursively discover tests in the directory.
    std::error_code ec;
    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(path, ec)) {
      if (ec)
        return AsyncOptionalTest::createReady(runtime, std::nullopt);

      if (entry.is_directory(ec)) {
        asyncChildren.emplace_back(
            discoverTestsInDirectory(entry.path(), ignoreModules));
      } else if (!ignoreModules && Filesystem::isMojoSourceFile(entry.path())) {
        asyncChildren.emplace_back(discoverTestsInMojoSource(entry.path()));
      }
    }

    // If there are no children, we're done.
    if (asyncChildren.empty())
      return AsyncOptionalTest::createReady(runtime, std::nullopt);
    // If there is only one child, return it directly.
    if (asyncChildren.size() == 1)
      return std::move(asyncChildren.front());

    auto result = AsyncOptionalTest::allocate(runtime);
    LLCL::andThenAsyncMoving(
        asyncChildren,
        [path, hasSourcePackageChild, result = result.copy()](
            MutableArrayRef<AnyAsyncValueRef> asyncChildren) mutable {
          // Otherwise, await the children and extract out the discovered tests.
          std::vector<Test> children;

          // If there is a source package child, pull out the tests from it.
          if (hasSourcePackageChild) {
            auto &pkgChild = asyncChildren.front();
            if (pkgChild.isError())
              return std::move(result).setToError(pkgChild.takeDiagnostic());
            if (auto &test = pkgChild.get<std::optional<Test>>())
              test->children.swap(children);
            asyncChildren = asyncChildren.drop_front();
          }

          // Extract out tests from the children.
          for (AnyAsyncValueRef &asyncChild : asyncChildren) {
            if (asyncChild.isError())
              return std::move(result).setToError(asyncChild.takeDiagnostic());
            if (auto &test = asyncChild.get<std::optional<Test>>())
              children.emplace_back(std::move(*test));
          }
          if (children.empty())
            return std::move(result).emplace(std::nullopt);
          // If there is only one child, return it directly.
          if (children.size() == 1)
            return std::move(result).emplace(std::move(children.front()));

          // Build a suite for this directory. To keep tests in a stable order,
          // we sort them by their ID.
          llvm::sort(children, [](const Test &lhs, const Test &rhs) {
            return lhs.getTestID() < rhs.getTestID();
          });
          std::move(result).emplace(
              Test(TestID(path.string()), std::move(children)));
        });
    return std::move(result);
  }

  LLCL::Runtime &runtime;
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ArrayRef<std::string> additionalImportPaths;
};

ErrorOr<std::optional<Test>>
Test::discoverFromID(LLCL::Runtime &runtime, const TestID &testID,
                     ArrayRef<std::string> additionalImportPaths) {
  std::filesystem::path path = testID.getFilePath();

  // Check that the path actually exists.
  std::error_code ec;
  if (!std::filesystem::exists(path, ec))
    return std::nullopt;
  TestDiscovery testDiscovery(runtime, additionalImportPaths);

  // Check if the test specifies a specific suite within the path. In this
  // case the path should be some mojo source.
  std::optional<StringRef> testSuiteName = testID.getTestSuite();
  std::optional<StringRef> testName = testID.getTest();
  if (testSuiteName || testName) {
    // TODO: Support doc tests defined in jupyter notebooks.
    if (!Filesystem::isMojoSourceFile(path) &&
        !Filesystem::isMojoSourcePackagePath(path)) {
      return std::nullopt;
    }

    // Read the test suite from the mojo source.
    ErrorOr<std::optional<Test>> testSuite =
        testDiscovery.discoverTestsInMojoSourceSync(path,
                                                    testSuiteName.value_or(""));
    if (testSuite.isError() || !*testSuite || !testName)
      return testSuite;

    // Validate that the test exists within the suite.
    if (const Test *test = (*testSuite)->getChild(*testName))
      return *test;
    return std::nullopt;
  }

  // If not, the file path either refers to a test suite for a file or a
  // directory.
  if (std::filesystem::is_directory(path, ec))
    return awaitTest(testDiscovery.discoverTestsInDirectory(path));

  // The path is a mojo source file.
  if (Filesystem::isMojoSourceFile(path))
    return testDiscovery.discoverTestsInMojoSourceSync(path);

  // TODO: Support doc tests defined in jupyter notebooks.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Execution Utilities
//===----------------------------------------------------------------------===//

/// Create a temporary output file for the test executor.
static ErrorOr<TempFile> createTempOutputFile() {
  // Create a temporary file to capture the output of the invocation.
  ErrorOr<TempFile> outOrErr = TempFile::create("test-out-%%%%%%.txt");
  if (failed(outOrErr))
    return Error("could not create temporary file for test output");
  return std::move(*outOrErr);
}

/// Emit an initialization error for the given set of tests.
static std::vector<TestExecutionResult> emitTestInitError(ArrayRef<Test> tests,
                                                          const Twine &error) {
  std::vector<TestExecutionResult> results;
  results.emplace_back(
      TestExecutionResult::buildInitError(tests[0].getTestID(), error.str()));
  for (const Test &child : tests.drop_back())
    results.emplace_back(TestExecutionResult::buildSkip(child.getTestID()));
  return results;
}

/// Process the result of a test executor execution.
static TestExecutionResult
processTestExecutorResults(ArrayRef<Test> tests,
                           std::vector<TestExecutionResult> results) {
  auto emitError = [&](StringRef message) {
    if (tests.size() == 1)
      return TestExecutionResult::buildInitError(tests[0].getTestID(), message);
    return TestExecutionResult(
        TestExecutionResult::kExecutionError, tests[0].getTestID().withTest(""),
        std::chrono::milliseconds(0), emitTestInitError(tests, message));
  };

  // Check that we got the right number of results.
  if (results.size() != tests.size())
    return emitError("fatal error: test execution generated an unexpected "
                     "number of results");

  // Handle the case where we're executing a single test.
  if (results.size() == 1)
    return results.front();

  // Check that the results are in the expected order.
  std::chrono::milliseconds duration = std::chrono::milliseconds(0);
  for (size_t i = 0, e = tests.size(); i < e; ++i) {
    if (results[i].getTestID() != tests[i].getTestID())
      return emitError(
          "fatal error: test execution generated unexpected results");
    duration += results[i].getDuration();
  }

  // Return an execution result for the parent suite, with the children results.
  bool hasError = results.back().getKind() != TestExecutionResult::kSuccess;
  return TestExecutionResult(hasError ? TestExecutionResult::kExecutionError
                                      : TestExecutionResult::kSuccess,
                             tests[0].getTestID().withTest(""), duration,
                             std::move(results));
}

namespace {
/// This class defines a simple server for communicating with the test executor.
struct TestServer {
  TestServer(MessageHandler &messageHandler) {
    messageHandler.notification("execution/result", this,
                                &TestServer::onExecutionResult);
  }

  void onExecutionResult(const TestExecutionResult &result) {
    results.emplace_back(result);
  }

  std::vector<TestExecutionResult> results;
};

/// This class represents a single execution instance, containing all of the
/// state related to invoking the test executor.
struct TestExecutionInstance {
  TestExecutionInstance(ArrayRef<Test> tests, TempFile outputTempFile,
                        std::FILE *outFile, llvm::sys::ProcessInfo processInfo)
      : tests(tests), outputTempFile(std::move(outputTempFile)),
        outFile(outFile), transport(outFile, nullOS), messageHandler(transport),
        testServer(messageHandler), processInfo(processInfo) {}
  ~TestExecutionInstance() { fclose(outFile); }

  /// Check the execution of the test executor. If the execution is
  /// complete, this will return the result of the execution. Otherwise,
  /// returns nullopt.
  std::optional<TestExecutionResult> checkExecution();

  /// The tests being executed.
  SmallVector<Test> tests;

  /// The temporary file used to capture the output of the test executor.
  std::optional<TempFile> outputTempFile;
  std::FILE *outFile;

  /// The various transport used in communicated with the test executor.
  llvm::raw_null_ostream nullOS;
  JSONTransport transport;
  MessageHandler messageHandler;
  TestServer testServer;

  /// The active process information for the test executor.
  llvm::sys::ProcessInfo processInfo;
};

/// This class represents a potentially resolved execution result.
class MaybeResolvedResult {
public:
  MaybeResolvedResult(
      llvm::unique_function<std::optional<TestExecutionResult>()> &&resolveFn)
      : resolveFn(std::move(resolveFn)) {}
  MaybeResolvedResult(TestExecutionResult &&result)
      : result(std::move(result)) {}

  /// Resolve the current result, returning success if the result is resolved,
  /// failure otherwise.
  LogicalResult resolve() {
    if (!result) {
      if (std::optional<TestExecutionResult> resolved = resolveFn())
        result = std::move(*resolved);
    }
    return success(result.has_value());
  }

  /// Take the resolved result. This asserts that the result is resolved.
  TestExecutionResult takeResolvedResult() {
    assert(result && "result is not resolved");
    return std::move(*result);
  }

private:
  llvm::unique_function<std::optional<TestExecutionResult>()> resolveFn;
  std::optional<TestExecutionResult> result;
};
} // namespace

std::optional<TestExecutionResult> TestExecutionInstance::checkExecution() {
  auto runTransport = [&] {
    // Run the main loop of the transport.
    if (llvm::Error error = transport.run(messageHandler)) {
      llvm::consumeError(std::move(error));

      // Clear the error state of the file, we can only read parts of the file
      // at a time, so we don't care about feof.
      clearerr(outFile);
    }
  };

  // Run the transport once to process any new state, and then check if the
  // process has completed.
  runTransport();
  auto resultInfo = llvm::sys::Wait(processInfo, /*SecondsToWait=*/0);
  if (!(resultInfo.ReturnCode || resultInfo.Pid))
    return std::nullopt;

  // If the process completed, flush out the transport one more time before
  // processing results.
  runTransport();
  return processTestExecutorResults(tests, std::move(testServer.results));
}

//===----------------------------------------------------------------------===//
// Test
//===----------------------------------------------------------------------===//

/// Execute the given set of tests, returning the result.
static MaybeResolvedResult
executeTests(ArrayRef<Test> tests,
             ArrayRef<std::string> additionalImportPaths) {
  auto emitInitError = [&](const Twine &error) {
    return processTestExecutorResults(tests, emitTestInitError(tests, error));
  };

  // Grab the path to the test executor.
  ErrorOr<KGEN::MojoConfig> config = KGEN::MojoConfig::open();
  if (config.isError())
    return emitInitError("unable to open Mojo configuration file: " +
                         Twine(config.getError()));
  StringRef testExecutorPath = config->getTestExecutorPath();

  // Create a input file for the test executor and write the set of tests to
  // execute to the input file.
  auto inFileOr = createTempOutputFile();
  if (failed(inFileOr))
    return emitInitError(inFileOr.getError());
  {
    llvm::raw_fd_ostream inOS(inFileOr->getFD(), /*shouldClose=*/false);
    llvm::json::OStream(inOS).value(llvm::json::Array(tests));
  }

  // Create a temporary output file for the test executor.
  auto outFileOr = createTempOutputFile();
  if (failed(outFileOr))
    return emitInitError(outFileOr.getError());
#ifndef _WIN32_
  std::FILE *outFile = fdopen(dup(outFileOr->getFD()), "r");
#else
  std::FILE *outFile = fdopen(_dup(outFileOr->getFD()), "r");
#endif

  // Invoke the test executor with the test ID, directing its output to the
  // file.
  std::string in = inFileOr->getPath().string();
  std::string out = outFileOr->getPath().string();
  const std::optional<StringRef> redirects[] = {
      /*stdin=*/in,
      /*stdout=*/out,
      /*stderr=*/std::nullopt,
  };
  SmallVector<StringRef> args = {testExecutorPath};
  for (StringRef path : additionalImportPaths)
    llvm::append_range(args, ArrayRef<StringRef>{"-I", path});

  std::vector<StringRef> env = getEnv();
  env.emplace_back("MODULAR_TELEMETRY_ENABLED=false");

  auto processInfo =
      llvm::sys::ExecuteNoWait(testExecutorPath, args, env, redirects);

  // Build an unresolved result that waits for the process to complete.
  auto instance = std::make_unique<TestExecutionInstance>(
      tests, std::move(*outFileOr), outFile, processInfo);
  return MaybeResolvedResult([instance = std::move(instance)]() {
    return instance->checkExecution();
  });
}

/// Execute the given doc test, returning the result.
static MaybeResolvedResult
executeDocTest(LLCL::Runtime &runtime, const Test &test,
               ArrayRef<std::string> additionalImportPaths) {
  // Doc tests are unique compare to unit tests in that they are execution
  // dependent on the previous tests in the same suite. As a result, we need to
  // execute all of the previous tests in the suite together with `test`.
  size_t index;
  if (test.getTestID().getTest()->getAsInteger(10, index)) {
    return TestExecutionResult::buildInitError(
        test.getTestID(), "id does not correspond to a valid doc test");
  }
  // If this is the first test, execute it directly.
  if (index == 0)
    return executeTests(test, additionalImportPaths);

  // Pull in the parent doc test suite.
  ErrorOr<std::optional<Test>> parentTestOr = Test::discoverFromID(
      runtime, test.getTestID().withTest(""), additionalImportPaths);
  if (parentTestOr || !*parentTestOr ||
      (**parentTestOr).getChildren().size() <= index)
    return TestExecutionResult::buildInitError(
        test.getTestID(), "id does not correspond to a valid doc test");
  const Test &parentTest = **parentTestOr;
  return executeTests(parentTest.getChildren().take_front(index + 1),
                      additionalImportPaths);
}

/// Execute the given test or suite, returning the result.
static MaybeResolvedResult
executeTestOrSuite(LLCL::Runtime &runtime, const Test &test,
                   ArrayRef<std::string> additionalImportPaths) {
  // If this is a test, execute it directly.
  const TestID &testID = test.getTestID();
  if (testID.getTest()) {
    if (testID.getTestSuite() && testID.getTestSuite()->ends_with("__doc__"))
      return executeDocTest(runtime, test, additionalImportPaths);
    return executeTests(test, additionalImportPaths);
  }
  // If this is a doc test suite, we can execute all of the tests together
  // (given that doc tests have execution dependent on the previous tests in the
  // same suite).
  if (testID.getTestSuite() && testID.getTestSuite()->ends_with("__doc__"))
    return executeTests(test.getChildren(), additionalImportPaths);

  // Otherwise, this is a suite. Execute each of the children, and collect the
  // results.
  std::vector<MaybeResolvedResult> results;
  for (const Test &child : test.getChildren()) {
    results.push_back(
        executeTestOrSuite(runtime, child, additionalImportPaths));
  }

  auto now = std::chrono::steady_clock::now();
  auto resolveFn = [&, now = now, results = std::move(results)]() mutable {
    bool allResolved = true;
    for (MaybeResolvedResult &result : results)
      allResolved &= succeeded(result.resolve());
    if (!allResolved)
      return std::optional<TestExecutionResult>();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - now);

    // If everything is resolved, we can build the final result.
    std::vector<TestExecutionResult> resolvedResults;
    TestExecutionResult::Kind suiteKind = TestExecutionResult::kSuccess;
    for (MaybeResolvedResult &maybeResult : results) {
      TestExecutionResult result = maybeResult.takeResolvedResult();
      if (result.getKind() < suiteKind)
        suiteKind = result.getKind();
      resolvedResults.emplace_back(std::move(result));
    }
    return std::make_optional(TestExecutionResult(suiteKind, testID, duration,
                                                  std::move(resolvedResults)));
  };
  return MaybeResolvedResult(std::move(resolveFn));
}

TestExecutionResult
Test::execute(LLCL::Runtime &runtime,
              ArrayRef<std::string> additionalImportPaths) const {
  // Execute this test and wait for it to resolve. We don't block here because
  // resolution of the result may involve communicating with multiple
  // test-executor processes.
  MaybeResolvedResult result =
      executeTestOrSuite(runtime, *this, additionalImportPaths);
  while (failed(result.resolve()))
    ;
  return result.takeResolvedResult();
}

//===----------------------------------------------------------------------===//
// JSON Serialization

bool KGEN::Mojo::fromJSON(const llvm::json::Value &value, Test &result,
                          llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  if (!o)
    return false;

  Test::SourceRange location;
  if (o.mapOptional("location", location))
    result.location = std::make_optional(location);
  return o.mapOptional("children", result.children) &&
         o.map("id", result.testID);
}

bool KGEN::Mojo::fromJSON(const llvm::json::Value &value,
                          Test::SourceRange &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("startLine", result.startLine) &&
         o.map("startColumn", result.startColumn) &&
         o.map("endLine", result.endLine) &&
         o.map("endColumn", result.endColumn);
}

llvm::json::Value KGEN::Mojo::toJSON(const Test &value) {
  llvm::json::Object object{{"id", value.testID}};
  if (!value.children.empty())
    object["children"] = llvm::json::Value(value.children);
  if (value.location)
    object["location"] = llvm::json::Value(*value.location);
  return std::move(object);
}

llvm::json::Value KGEN::Mojo::toJSON(const Test::SourceRange &value) {
  return llvm::json::Object{
      {"startLine", value.startLine},
      {"startColumn", value.startColumn},
      {"endLine", value.endLine},
      {"endColumn", value.endColumn},
  };
}
