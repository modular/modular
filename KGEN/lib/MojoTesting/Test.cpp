//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTesting/Test.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
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

raw_ostream &KGEN::Mojo::operator<<(raw_ostream &os, const TestID &testID) {
  return os << testID.strref();
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
  static TestID getDeclTestID(StringRef path, Operation *op) {
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

    // Collect the symbols making up the test name.
    SmallVector<std::string> symbols;
    do {
      if (definesTestSuite(symOp))
        break;
      symbols.push_back(printSymbol(symOp.getNameAttr()));
    } while ((symOp = symOp->getParentOfType<mlir::SymbolOpInterface>()));
    std::string testName = llvm::join(llvm::reverse(symbols), ".");

    // Collect the symbols making up the test suite name.
    symbols.clear();
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

  //===--------------------------------------------------------------------===//
  // Discover: Mojo Source

  /// Discover tests defined within the given Mojo decl.
  static void discoverTestsInDecl(StringRef path, MojoASTDeclRef ref,
                                  SharedState &shared, std::vector<Test> &tests,
                                  bool processUnitTests) {
    // Check if the decl defines a unit test.
    if (processUnitTests && doesDeclDefineUnitTest(ref, shared))
      tests.emplace_back(Test(getDeclTestID(path, ref)));

    for (const MojoASTDeclRef::ChildEntry &child : ref.getChildren()) {
      for (MojoASTDeclRef decl : child.getDecls()) {
        // We only process direct children.
        if (decl->getParentDecl() != &*ref)
          continue;

        // Check if the decl defines a new test suite.
        if (definesTestSuite(decl.getIfOperation())) {
          std::vector<Test> children;
          discoverTestsInDecl(path, decl, shared, children, processUnitTests);
          if (!children.empty()) {
            tests.emplace_back(
                Test(getDeclTestID(path, decl), std::move(children)));
          }
        } else {
          discoverTestsInDecl(path, decl, shared, tests, processUnitTests);
        }
      }
    }
  }

  std::optional<Test>
  discoverTestsInMojoSource(const std::filesystem::path &path) {
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

    // If the decl doesn't define a test suite, we're done.
    if (!definesTestSuite(moduleDecl.getIfOperation()))
      return std::nullopt;
    // We only process unit tests for certain paths.
    std::string pathStem = path.stem().string();
    bool processUnitTests = StringRef(pathStem).starts_with("test_") ||
                            StringRef(pathStem).ends_with("_test");

    // Process the decl to discover tests.
    std::vector<Test> tests;
    discoverTestsInDecl(path.string(), moduleDecl,
                        parserContext.getSharedState(), tests,
                        processUnitTests);
    if (tests.empty())
      return std::nullopt;
    return Test(TestID(path.string()), std::move(tests));
  }

  //===--------------------------------------------------------------------===//
  // Discover: FileSystem

  /// Discovers tests defined within in the given directory.
  std::optional<Test>
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
        if (auto child = discoverTestsInDirectory(entry.path()))
          children.emplace_back(std::move(*child));
      } else if (Filesystem::isMojoSourceFile(entry.path())) {
        if (auto child = discoverTestsInMojoSource(entry.path()))
          children.emplace_back(std::move(*child));
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

std::optional<Test> Test::discoverFromID(const TestID &testID) {
  std::filesystem::path path = testID.getFilePath();

  // Check that the path actually exists.
  std::error_code ec;
  if (!std::filesystem::exists(path, ec))
    return std::nullopt;

  // Check if the path is a directory.
  if (std::filesystem::is_directory(path, ec))
    return TestDiscovery().discoverTestsInDirectory(path);

  // The path is a mojo source file.
  if (Filesystem::isMojoSourceFile(path))
    return TestDiscovery().discoverTestsInMojoSource(path);

  // TODO: Support doc tests defined in jupyter notebooks.
  return std::nullopt;
}
