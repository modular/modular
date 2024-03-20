//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTESTING_TEST_H
#define KGEN_MOJOTESTING_TEST_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include <filesystem>

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

private:
  /// The id of test.
  std::string id;

  /// The individual components of the test ID.
  StringRef path, testSuite, test;
};
raw_ostream &operator<<(raw_ostream &os, const TestID &testID);

//===----------------------------------------------------------------------===//
// Test
//===----------------------------------------------------------------------===//

/// This class represents a test or test suite.
class Test {
public:
  Test() = default;

  //===--------------------------------------------------------------------===//
  // Accessors

  /// Return the ID of the test.
  const TestID &getTestID() const { return testID; }

  /// Return the child tests nested within this test.
  ArrayRef<Test> getChildren() const { return children; }

  //===--------------------------------------------------------------------===//
  // Discovery

  /// Discover the test structure from the given Test ID. Returns nullopt if no
  /// test or suite was discovered.
  static std::optional<Test> discoverFromID(const TestID &testID);

  //===--------------------------------------------------------------------===//
  // Display

  /// Print the test and its children to the given stream.
  void print(raw_ostream &os) const;

private:
  Test(TestID testID, std::vector<Test> children = {})
      : testID(std::move(testID)), children(std::move(children)) {}

  /// A utility class used in the implementation of test discovery.
  struct TestDiscovery;

  /// The id of test.
  TestID testID;

  /// The nested tests of the test.
  std::vector<Test> children;
};
raw_ostream &operator<<(raw_ostream &os, const Test &test);
} // namespace M::KGEN::Mojo

#endif // KGEN_MOJOTESTING_TEST_H
