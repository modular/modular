//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Configuration.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

#ifdef LLVM_ON_UNIX
#include <sys/mman.h>
#endif // LLVM_ON_UNIX

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

TEST(Configuration, EnvOverride) {
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
  EXPECT_EQ(cfg.getValueOr("only.default.value", "default"), "default");
}

TEST(Configuration, DisabledOverride) {
  // Check that the env override works as expected.
  setenv("MODULAR_AKEY", "foo", 0);
  setenv("MODULAR_SECTION_SUBSECTION_KEY3", "bar", 0);

  auto unsetEnv = llvm::make_scope_exit([]() {
    unsetenv("MODULAR_AKEY");
    unsetenv("MODULAR_SECTION_SUBSECTION_KEY3");
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

  cfg.setEnvOverride(false);

  EXPECT_EQ(cfg.getValue("akey"), "value");
  EXPECT_EQ(cfg.getValue("section.key2"), "value2");
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "value3");
  EXPECT_EQ(cfg.getValueOr("only.default.value", "default"), "default");

  // Can be re-enabled for the default behavior.
  cfg.setEnvOverride(true);
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "bar");
  setenv("MODULAR_SECTION_SUBSECTION_KEY3", "newbar", 1);
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "newbar");

  // Once it's overwritten, it's overwritten forever.
  cfg.setEnvOverride(false);
  setenv("MODULAR_SECTION_SUBSECTION_KEY3", "kombuchabar", 1);
  EXPECT_EQ(cfg.getValue("section.subsection.key3"), "newbar");
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

#ifdef LLVM_ON_UNIX
// Regression test for a bug in Configuration
TEST(Configuration, PageBoundary) {
  auto pageSize = llvm::cantFail(llvm::sys::Process::getPageSize());
  std::error_code ec;
  llvm::sys::OwningMemoryBlock block(llvm::sys::Memory::allocateMappedMemory(
      2 * pageSize, nullptr,
      llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE, ec));
  ASSERT_FALSE(ec) << "Failed to allocate memory block: " << ec.message();
  // Can't use llvm::sys::Memory::protectMappedMemory because it considers
  // PFlags = 0 to be invalid.
  int mprotectResult = ::mprotect(
      reinterpret_cast<char *>(block.base()) + pageSize, pageSize, PROT_NONE);
  ASSERT_NE(mprotectResult, -1)
      << std::error_code(errno, std::generic_category()).message();
  // Note: No trailing newline!
  StringRef content("key = value");
  char *contentPlacePtr =
      reinterpret_cast<char *>(block.base()) + pageSize - content.size();
  StringRef contentInPlace(contentPlacePtr, content.size());
  memcpy(contentPlacePtr, content.data(), content.size());
  ASSERT_EQ(content, contentInPlace);
  // OK, now try to parse!
  Config cfg;
  auto err = cfg.parseFrom(contentInPlace);
  ASSERT_FALSE(err.isError()) << err.getError();
  EXPECT_EQ(cfg.getValue("key"), "value");
}
#endif // LLVM_ON_UNIX

TEST(Configuration, GetValuesInSection) {
  StringRef input = R"(
key = value
#maybe a comment in between these guys
key4 = value4

# this is a comment
[section]
key2 = value2 # with a comment
; another comment
key3 = value3

[section.withdot]
key5 = value5
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  SmallVector<std::pair<StringRef, StringRef>> globals;
  cfg.getValuesInSection("", globals);
  EXPECT_EQ(globals.size(), 2u);
  EXPECT_EQ(globals[0].first, "key") << std::string(globals[0].first);
  EXPECT_EQ(globals[0].second, "value") << std::string(globals[0].second);
  EXPECT_EQ(globals[1].first, "key4") << std::string(globals[1].first);
  EXPECT_EQ(globals[1].second, "value4") << std::string(globals[1].second);

  SmallVector<std::pair<StringRef, StringRef>> section;
  cfg.getValuesInSection("section", section);
  EXPECT_EQ(section.size(), 2u);
  EXPECT_EQ(section[0].first, "key2") << std::string(section[0].first);
  EXPECT_EQ(section[0].second, "value2") << std::string(section[0].second);
  EXPECT_EQ(section[1].first, "key3") << std::string(section[1].first);
  EXPECT_EQ(section[1].second, "value3") << std::string(section[1].second);

  SmallVector<std::pair<StringRef, StringRef>> withdot;
  cfg.getValuesInSection("section.withdot", withdot);
  EXPECT_EQ(withdot.size(), 1u);
  EXPECT_EQ(withdot[0].first, "key5") << std::string(withdot[0].first);
  EXPECT_EQ(withdot[0].second, "value5") << std::string(withdot[0].second);
}

TEST(Configuration, GetAllValues) {
  StringRef input = R"(
key = value
#maybe a comment in between these guys
key4 = value4

# this is a comment
[section]
key2 = value2 # with a comment
; another comment
key3 = value
)";

  Config cfg;
  auto err = cfg.parseFrom(input);
  ASSERT_FALSE(err.isError()) << err.getError();

  const llvm::StringMap<std::string> &allVals = cfg.getAllValues();
  EXPECT_TRUE(allVals.contains("key"));
  EXPECT_TRUE(allVals.contains("key4"));
  EXPECT_EQ(allVals.at("key"), "value");
  EXPECT_TRUE(allVals.contains("section.key2"));
  EXPECT_EQ(allVals.at("section.key3"), "value");
}

TEST(Configuration, Merge) {
  StringRef one = R"(
key = value
#maybe a comment in between these guys
key4 = value4
)";

  StringRef two = R"(
# this is a comment
[section]
key2 = value2 # with a comment
; another comment
key3 = value3

[section.withdot]
key5 = value5
)";

  Config from;
  auto err = from.parseFrom(one);
  ASSERT_FALSE(err.isError()) << err.getError();

  Config to;
  err = to.parseFrom(two);
  ASSERT_FALSE(err.isError()) << err.getError();

  err = to.copyFrom(from);
  ASSERT_FALSE(err.isError()) << err.getError();

  EXPECT_EQ(to.getValue("key"), "value");

  err = from.copyFrom(from);
  ASSERT_TRUE(err.getError());
}

TEST(Configuration, BooleanValues) {
  using R = ErrorOr<bool>;
  Config cfg;
  EXPECT_EQ(R(false), cfg.getValueAsBool("example", false));
  EXPECT_EQ(R(true), cfg.getValueAsBool("example", true));
  for (auto value : {"0", "false", "no", "FaLsE"}) {
    cfg.setValue("example", value);
    EXPECT_EQ(R(false), cfg.getValueAsBool("example", false));
    EXPECT_EQ(R(false), cfg.getValueAsBool("example", true));
  }
  for (auto value : {"1", "true", "yes", "TrUe"}) {
    cfg.setValue("example", value);
    EXPECT_EQ(R(true), cfg.getValueAsBool("example", false));
    EXPECT_EQ(R(true), cfg.getValueAsBool("example", true));
  }
  cfg.setValue("example", "maybe");
  auto result = cfg.getValueAsBool("example", false);
  EXPECT_TRUE(result.isError());
  EXPECT_STREQ("Unable to interpret configuration key 'example' with value "
               "'maybe' as boolean",
               result.getError());
}

TEST(Configuration, Override) {
  StringRef one = R"(
global_key_1 = value1
#maybe a comment in between these guys
global_key_2 = value2

[section.one]
key_1 = value1
key_2 = value2

[sectionTwo]
key_1 = value1
key_2 = value2
)";

  StringRef two = R"(
# this is a comment
global_key_2 = value2_override

[section.one]
key_1 = value1_override

[section3]
key_1 = value1_override # with a comment
key_2 = value2_override
key_3 = value3_override
)";

  Config base;
  auto err = base.parseFrom(one);
  ASSERT_FALSE(err.isError()) << err.getError();
  ASSERT_EQ(base.getAllValues().size(), (size_t)6);

  Config over;
  err = over.parseFrom(two);
  ASSERT_FALSE(err.isError()) << err.getError();
  ASSERT_EQ(over.getAllValues().size(), (size_t)5);

  err = base.overrideFrom(over);
  ASSERT_FALSE(err.isError()) << err.getError();
  ASSERT_EQ(base.getAllValues().size(), (size_t)8);

  EXPECT_EQ(base.getValue("global_key_1"), "value1");
  EXPECT_EQ(base.getValue("global_key_2"), "value2_override");

  // section.one is entirely overridden
  EXPECT_EQ(base.getValue("section.one.key_1"), "value1_override");
  // key_2 is not present in over. it should not present
  EXPECT_EQ(base.getValueOr("section.one.key_2", "UNSET"), "UNSET");

  // these should be unaffected by overrideFrom
  EXPECT_EQ(base.getValue("sectionTwo.key_1"), "value1");
  EXPECT_EQ(base.getValue("sectionTwo.key_2"), "value2");

  // this section comes from over
  EXPECT_EQ(base.getValue("section3.key_1"), "value1_override");
  EXPECT_EQ(base.getValue("section3.key_2"), "value2_override");
  EXPECT_EQ(base.getValue("section3.key_3"), "value3_override");
}
