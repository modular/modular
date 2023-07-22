//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Configuration.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace M;

/// Get a source manager for a given buffer so we can check diagnostics.
static llvm::SourceMgr getSourceMgr(StringRef buffer) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(buffer);
  llvm::SourceMgr mgr;
  (void)mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());
  return mgr;
}

TEST(Configuration, SectionParse) {
  StringRef input = R"(
[section]
key = value
# this is a comment
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  EXPECT_EQ(cfg.getValue("section.key"), "value");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "value3");
}

TEST(Configuration, RoundTrip) {
  StringRef input = R"(
[section]
key = value
# this is a comment
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";
  StringRef output = R"([section]
key = value
key2 = value2
new_key = new value

[section.subsection]
key3 = value3

)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  cfg.setValue("section.new_key", "new value");

  std::string out;
  llvm::raw_string_ostream stream(out);
  cfg.flush(stream);

  EXPECT_EQ(out, output);
}

TEST(Configuration, Globals) {
  StringRef input = R"(
key = value
# this is a comment
[section]
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  EXPECT_EQ(cfg.getValue("key"), "value");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "value3");
}

TEST(Configuration, Override) {
  // Check that the env override works as expected.
  setenv("MODULAR_AKEY", "foo", 0);
  setenv("MODULAR_SECTION_SUBSECTION_KEY3", "bar", 0);

  auto unsetEnv = llvm::make_scope_exit([]() {
    unsetenv("MODULAR_KEY");
    unsetenv("MODULAR_SECTION_SUBSECTION_KEY");
  });

  StringRef input = R"(
akey = value
# this is a comment
[section]
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  EXPECT_EQ(cfg.getValue("akey"), "foo");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "bar");
}

TEST(Configuration, SetValue) {
  StringRef input = R"(
akey = value
# this is a comment
[section]
key2 = value2 # with a comment
; another comment

[section.subsection] # yet another comment
key3 = value3
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  cfg.setValue("akey", "foo");
  EXPECT_EQ(cfg.getValue("akey"), "foo");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  cfg.setValue("section.subsection.key3", "bar");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "bar");
}

TEST(Configuration, MalformedLine) {
  StringRef input = R"(
[section]
key = value
# this is a comment
key2 = value2 # with a comment
; another comment
malformed line here

[section.subsection] # yet another comment
key3 = value3
)";
  llvm::SourceMgr mgr = getSourceMgr(input);

  Config cfg;
  auto err = cfg.parseFrom(input, &mgr);
  ASSERT_TRUE(err.isError());
  EXPECT_EQ(StringRef(err.getError()),
            StringRef(R"(error: malformed line: expected `key = value`
malformed line here
^
)"));
}
