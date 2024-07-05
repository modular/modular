//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTESTING_TEST_H
#define KGEN_MOJOTESTING_TEST_H

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include <chrono>
#include <filesystem>

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN::Mojo {
//===----------------------------------------------------------------------===//
// TestID
//===----------------------------------------------------------------------===//

/// This class represents an identifier for a test or test suite. A TestID is
/// some form of `<path>(@<test-suite>)?(::<test>)?`:
///   - `<path>` is the file path of the test.
///   - `<test-suite>` is the name of the test suite that the test is
///     associated with within the path.
///   - `<test>` is the name of the test within the test suite.
class TestID {
public:
  TestID() = default;
  TestID(TestID &&rhs) = default;
  TestID(const Twine &argID);
  TestID(StringRef pathArg, StringRef testSuiteArg, StringRef testArg = {});
  TestID(const TestID &rhs) { *this = rhs; }
  TestID &operator=(const TestID &rhs);
  TestID &operator=(TestID &&rhs) = default;

  /// Compare this id with another.
  bool operator==(const TestID &rhs) const { return strref() == rhs.strref(); }
  bool operator!=(const TestID &rhs) const { return strref() != rhs.strref(); }
  bool operator<(const TestID &rhs) const {
    return strref().compare_numeric(rhs.strref()) < 0;
  }

  /// Return the full encoded ID.
  StringRef strref() const { return id; }

  /// Return the file path of the test.
  std::filesystem::path getFilePath() const;

  /// Return the test suite component of the test ID, if present.
  std::optional<StringRef> getTestSuite() const {
    return testSuite.empty() ? std::nullopt
                             : std::make_optional(StringRef(testSuite));
  }

  /// Return the test component of the test ID, if present.
  std::optional<StringRef> getTest() const {
    return test.empty() ? std::nullopt : std::make_optional(StringRef(test));
  }

  /// Return a new TestID using the path and test suite of this ID, but with
  /// the provided test name.
  TestID withTest(StringRef test) const;

  /// Parse a scoped name, that is used within a test ID (either as the test
  /// suite or test component), into the set of scopes it defines.
  static ErrorOr<SmallVector<std::string>>
  parseScopedName(StringRef scopedName);

private:
  friend bool fromJSON(const llvm::json::Value &value, TestID &result,
                       llvm::json::Path path);
  friend llvm::json::Value toJSON(const TestID &value);

  /// The id of test.
  std::string id;

  /// The individual components of the test ID.
  std::string path, testSuite, test;
};
raw_ostream &operator<<(raw_ostream &os, const TestID &testID);

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, TestID &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const TestID &value);

//===----------------------------------------------------------------------===//
// TestExecutionResult
//===----------------------------------------------------------------------===//

/// This class represents the result of executing a test.
class TestExecutionResult {
public:
  enum Kind {
    kInitializationError,
    kExecutionError,
    kSuccess,
    kSkipped,
  };

  TestExecutionResult() = default;
  TestExecutionResult(
      Kind kind, TestID testID,
      std::chrono::milliseconds duration = std::chrono::milliseconds(0),
      std::string error = "", std::string stdOut = "", std::string stdErr = "")
      : kind(kind), testID(std::move(testID)), duration(duration),
        error(std::move(error)), stdOut(std::move(stdOut)),
        stdErr(std::move(stdErr)) {}
  TestExecutionResult(Kind kind, TestID testID,
                      std::chrono::milliseconds duration,
                      std::vector<TestExecutionResult> children)
      : kind(kind), testID(std::move(testID)), duration(duration),
        children(std::move(children)) {}

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  /// Return the result kind.
  Kind getKind() const { return kind; }

  /// Return the ID of the test that was executed.
  const TestID &getTestID() const { return testID; }

  /// Return the duration of the test execution.
  std::chrono::milliseconds getDuration() const { return duration; }

  /// Return the various outputs of the test.
  StringRef getError() const { return error; }
  StringRef getStdOut() const { return stdOut; }
  StringRef getStdErr() const { return stdErr; }

  /// Return the children results.
  ArrayRef<TestExecutionResult> getChildren() const { return children; }

  //===--------------------------------------------------------------------===//
  // Construction
  //===--------------------------------------------------------------------===//

  /// Return an initialization error result.
  static TestExecutionResult buildInitError(TestID testID, StringRef error) {
    return {kInitializationError, std::move(testID),
            std::chrono::milliseconds(0), error.str()};
  }

  /// Return a skip result.
  static TestExecutionResult buildSkip(TestID testID) {
    return {kSkipped, std::move(testID)};
  }

  //===--------------------------------------------------------------------===//
  // Display
  //===--------------------------------------------------------------------===//

  /// Print the result in a human readable form to the given stream.
  void print(raw_ostream &os) const;

private:
  friend bool fromJSON(const llvm::json::Value &value,
                       TestExecutionResult &result, llvm::json::Path path);
  friend llvm::json::Value toJSON(const TestExecutionResult &value);

  /// The result of the test.
  Kind kind = kSuccess;

  /// The ID of the test that was executed.
  TestID testID;

  /// The duration of the execution of this test.
  std::chrono::milliseconds duration = std::chrono::milliseconds(0);

  /// The error emitted when executing this test, if any.
  std::string error;

  /// The output of the test when executed.
  std::string stdOut;
  std::string stdErr;

  /// The nested results of the execution.
  std::vector<TestExecutionResult> children;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, TestExecutionResult &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const TestExecutionResult &value);

//===----------------------------------------------------------------------===//
// Test
//===----------------------------------------------------------------------===//

/// This class represents a test or test suite.
class Test {
public:
  /// A simple range location within a source file. A line or column of '0'
  /// represents an invalid location.
  struct SourceRange {
    SourceRange(int startLine = 0, int startColumn = 0, int endLine = 0,
                int endColumn = 0)
        : startLine(startLine), startColumn(startColumn), endLine(endLine),
          endColumn(endColumn) {}

    int startLine, startColumn;
    int endLine, endColumn;
  };

  Test() = default;

  //===--------------------------------------------------------------------===//
  // Accessors

  /// Return the ID of the test.
  const TestID &getTestID() const { return testID; }

  /// Return the child tests nested within this test.
  ArrayRef<Test> getChildren() const { return children; }

  /// Return the child referenced by the given ID, nullptr if not found.
  const Test *getChild(StringRef test) const {
    auto it = childrenMap.find(testID.withTest(test).strref());
    return it == childrenMap.end() ? nullptr : &children[it->second];
  }

  /// Return the source range of the test within the source file, if present.
  std::optional<SourceRange> getSourceRange() const { return location; }

  //===--------------------------------------------------------------------===//
  // Discovery

  /// Discover the test structure from the given Test ID. Returns nullopt if no
  /// test or suite was discovered, or error if an error occurred.
  /// `additionalImportPaths` is a list of additional include paths to use when
  /// resolving mojo imports.
  static ErrorOr<std::optional<Test>>
  discoverFromID(LLCL::Runtime &runtime, const TestID &testID,
                 ArrayRef<std::string> additionalImportPaths);

  //===--------------------------------------------------------------------===//
  // Display

  /// Print the test and its children to the given stream.
  void print(raw_ostream &os) const;

  //===--------------------------------------------------------------------===//
  // Execution

  /// Execute the test, returning all of the collected results.
  /// `additionalImportPaths` is a list of additional include paths to use when
  /// resolving mojo imports.
  TestExecutionResult
  execute(LLCL::Runtime &runtime,
          ArrayRef<std::string> additionalImportPaths) const;

private:
  friend bool fromJSON(const llvm::json::Value &value, Test &result,
                       llvm::json::Path path);
  friend llvm::json::Value toJSON(const Test &value);

  Test(TestID testID, std::vector<Test> newChildren = {},
       std::optional<SourceRange> location = {})
      : testID(std::move(testID)), children(std::move(newChildren)),
        location(location) {
    for (unsigned i = 0, e = children.size(); i != e; ++i)
      childrenMap[children[i].getTestID().strref()] = i;
  }

  /// A utility class used in the implementation of test discovery.
  struct TestDiscovery;

  /// The id of test.
  TestID testID;

  /// The nested tests of the test.
  std::vector<Test> children;
  llvm::StringMap<unsigned> childrenMap;

  /// An optional location of the test within the source file.
  std::optional<SourceRange> location;
};
raw_ostream &operator<<(raw_ostream &os, const Test &test);

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, Test &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value, Test::SourceRange &result,
              llvm::json::Path path);
llvm::json::Value toJSON(const Test &value);
llvm::json::Value toJSON(const Test::SourceRange &value);
} // namespace M::KGEN::Mojo

#endif // KGEN_MOJOTESTING_TEST_H
