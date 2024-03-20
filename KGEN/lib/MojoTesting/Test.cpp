//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTesting/Test.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Filesystem/Paths.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include <filesystem>

using namespace M;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;

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
  path = strref().take_front(rhs.path.size());

  // Map the string references from the rhs to the new id.
  auto mapString = [&](StringRef str) {
    return StringRef(id.data() + (str.data() - rhs.id.data()), str.size());
  };
  if (!rhs.testSuite.empty())
    testSuite = mapString(rhs.testSuite);
  if (!rhs.test.empty())
    test = mapString(rhs.test);
  return *this;
}

std::filesystem::path TestID::getFilePath() const {
  std::error_code ec;
  return std::filesystem::weakly_canonical(path.str(), ec);
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

  return o.map("testID", result.testID) && o.map("error", result.error) &&
         o.map("stdErr", result.stdErr) && o.map("stdOut", result.stdOut);
}

llvm::json::Value KGEN::Mojo::toJSON(const TestExecutionResult &value) {
  return llvm::json::Object{
      {"kind", stringifyResultKind(value.kind)},
      {"duration_ms", value.duration.count()},
      {"testID", value.testID},
      {"error", value.error},
      {"stdErr", value.stdErr},
      {"stdOut", value.stdOut},
  };
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

/// This class is used to discover tests from within a specific Mojo file.
struct Test::TestDiscovery {
  TestDiscovery() {
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    ctx.appendDialectRegistry(registry);
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
      if (isa<ModuleOp>(symOp->getParentOp()))
        break;
      symbols.push_back(printSymbol(symOp.getNameAttr()));
    } while ((symOp = symOp->getParentOfType<mlir::SymbolOpInterface>()));
    std::string testSuiteName = llvm::join(llvm::reverse(symbols), ".");

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

    // Check for a test of the form: `fn test()`.
    if (resultType.isNoneType())
      return argTypes.empty();

    // Otherwise, check for a test of the form `def test()`. This form returns
    // an object and raises.
    return resultType.isEqualCanon(
               shared.lookupObjectType(context, context.getLoc())) &&
           fnSignature.isThrows() && argTypes.size() == 1;
  }

  static bool doesDeclDefineUnitTest(MojoASTDeclRef decl, SharedState &shared) {
    return doesDeclDefineUnitTest(decl->getIfOperation(), *decl, shared);
  }

  /// Return a doc test suite defined by the given decl, or nullopt if no doc
  /// tests are defined.
  static std::optional<Test> getDocTestSuiteFromDecl(StringRef filePath,
                                                     Operation *declOp) {
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

    std::vector<Test> children;
    for (size_t i : llvm::seq(docString.getCodeBlocks().size()))
      children.emplace_back(Test(testID.strref() + "::" + Twine(i)));
    return Test(testID, std::move(children));
  }
  static std::optional<Test> getDocTestSuiteFromDecl(StringRef filePath,
                                                     MojoASTDeclRef decl) {
    if (Operation *op = decl->getIfOperation())
      return getDocTestSuiteFromDecl(filePath, op);
    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // Discover: Mojo Source

  /// Discover tests defined by given Mojo decl.
  static void discoverTestsInDecl(StringRef path, MojoASTDeclRef ref,
                                  SharedState &shared, std::vector<Test> &tests,
                                  bool processUnitTests) {
    // Check if the decl defines a unit test.
    if (processUnitTests && doesDeclDefineUnitTest(ref, shared))
      tests.emplace_back(Test(getDeclTestID(path, ref)));

    // Check for a doc test suite if this isn't a package (package doc strings
    // are just a copy from their __init__).
    if (!isa<PackageOp>(*ref)) {
      if (std::optional<Test> test = getDocTestSuiteFromDecl(path, ref))
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

  ErrorOr<std::optional<Test>>
  discoverTestsInMojoSource(const std::filesystem::path &path,
                            StringRef suiteName = {}) {
    KGEN::CompilationOptions compilationOptions;
    ParserConfig parserConfig(&ctx, compilationOptions);

    // Process the mojo file, ignoring any diagnostics emitted along the way
    // (we don't care about emitting errors here, just discovering tests).
    llvm::SourceMgr sourceManager;
    sourceManager.setDiagHandler([](const llvm::SMDiagnostic &diag, void *) {});
    MojoParserContext parserContext(sourceManager, parserConfig);
    MojoASTDeclRef moduleDecl = parserContext.parseFileOrPackage(path);
    if (!moduleDecl || !moduleDecl.getIfOperation())
      return std::nullopt;

    // Process the case where we're looking for a specific test suite.
    MojoASTDeclRef decl = moduleDecl;
    if (!suiteName.empty()) {
      // If this is a doc test suite for the top-level decl, handle that
      // immediately.
      if (suiteName == "__doc__")
        return getDocTestSuiteFromDecl(path.string(), decl.getIfOperation());

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
        return getDocTestSuiteFromDecl(path.string(), op);
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

  /// Discovers tests defined within in the given directory.
  ErrorOr<std::optional<Test>>
  discoverTestsInDirectory(const std::filesystem::path &path) {
    // If the path is a mojo source package, we parse can directly parse out
    // the tests from the package.
    if (Filesystem::isMojoSourcePackagePath(path))
      return discoverTestsInMojoSource(path);

    // Otherwise, recursively discover tests in the directory.
    std::error_code ec;
    std::vector<Test> children;
    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(path, ec)) {
      if (ec)
        return std::nullopt;

      if (entry.is_directory(ec)) {
        auto child = discoverTestsInDirectory(entry.path());
        if (!child.isError() && *child)
          children.emplace_back(std::move(**child));
      } else if (Filesystem::isMojoSourceFile(entry.path())) {
        auto child = discoverTestsInMojoSource(entry.path());
        if (!child.isError() && *child)
          children.emplace_back(std::move(**child));
      }
    }

    // If there are no children, we're done.
    if (children.empty())
      return std::nullopt;
    // If there is only one child, return it directly.
    if (children.size() == 1)
      return std::move(children.front());
    // Otherwise, build a suite for this directory. To keep tests in a stable
    // order, we sort them by their ID.
    llvm::sort(children, [](const Test &lhs, const Test &rhs) {
      return lhs.getTestID() < rhs.getTestID();
    });
    return Test(TestID(path.string()), std::move(children));
  }

  mlir::MLIRContext ctx;
};

ErrorOr<std::optional<Test>> Test::discoverFromID(const TestID &testID) {
  std::filesystem::path path = testID.getFilePath();

  // Check that the path actually exists.
  std::error_code ec;
  if (!std::filesystem::exists(path, ec))
    return std::nullopt;

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
        TestDiscovery().discoverTestsInMojoSource(path,
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
    return TestDiscovery().discoverTestsInDirectory(path);

  // The path is a mojo source file.
  if (Filesystem::isMojoSourceFile(path))
    return TestDiscovery().discoverTestsInMojoSource(path);

  // TODO: Support doc tests defined in jupyter notebooks.
  return std::nullopt;
}
