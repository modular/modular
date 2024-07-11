//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Support/Resource.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::AsyncRT;

namespace {

TEST(Resource, Correct) {
  auto resource = Resource::allocate("test");
  auto use1 = resource->beginUse("use1");
  auto use2 = resource->beginUse("use2");
  use1.reset();
  use2.reset();
  resource->markFreed();
}

TEST(Resource, NestedReferences) {
  auto resource = Resource::allocate("test");
  auto use1 = resource->beginUse("use1", kReferencingResourceUse,
                                 ResourceSection(5, 15));
  auto use2 = resource->beginUse("use2", kReferencingResourceUse,
                                 ResourceSection(5, 15));
  auto use3 = resource->beginUse("use3", kReferencingResourceUse,
                                 ResourceSection(8, 12));
  auto use4 = resource->beginUse("use3", kReferencingResourceUse,
                                 ResourceSection(10, 12));
  use1.reset();
  use2.reset();
  use3.reset();
  use4.reset();
  resource->markFreed();
}

TEST(Resource, UseAfterFreeInFlight_ExpectDeath) {
  auto resource = Resource::allocate("test");
  auto use1 = resource->beginUse("use1");
  auto use2 = resource->beginUse("use2");
  use1.reset();
  EXPECT_DEATH_IF_SUPPORTED(
      resource->markFreed(),
      "attempting to free a resource while it still has references");
}

TEST(Resource, UseAfterFreeNew_ExpectDeath) {
  auto resource = Resource::allocate("test");
  auto use1 = resource->beginUse("use1");
  use1.reset();
  resource->markFreed();
  EXPECT_DEATH_IF_SUPPORTED(resource->beginUse("use2"),
                            "attempting to use a freed resource");
}

TEST(Resource, Uninitialized_ExpectDeath) {
  auto resource = Resource::allocate("test", /*isInitialized=*/false);
  EXPECT_DEATH_IF_SUPPORTED(
      resource->beginUse("use1", kReadingResourceUse),
      "attempting to read from uninitialized \\(section of\\) resource");
}

TEST(Resource, ReadWriteRace_ExpectDeath) {
  auto resource = Resource::allocate("test", /*isInitialized=*/true);
  auto use1 = resource->beginUse("read", kReadingResourceUse);
  EXPECT_DEATH_IF_SUPPORTED(
      resource->beginUse("write", kWritingResourceUse),
      "requested section for writing overlaps with existing section all for "
      "reading with active uses \\{'read'\\}");
}

TEST(Resource, WriteWriteRace_ExpectDeath) {
  auto resource = Resource::allocate("test", /*isInitialized=*/true);
  auto use1 = resource->beginUse("write1", kWritingResourceUse);
  EXPECT_DEATH_IF_SUPPORTED(
      resource->beginUse("write2", kWritingResourceUse),
      "requested section for writing overlaps with existing section all for "
      "writing with active uses \\{'write1'\\}");
}

TEST(Resource, ReadAftenUninit_ExpectDeath) {
  auto resource = Resource::allocate("test", /*isInitialized=*/true);
  auto use1 = resource->beginUse("read", kReadingResourceUse);
  EXPECT_DEATH_IF_SUPPORTED(resource->markUninitialized(use1),
                            "attempting to mark \\(section of) resource as "
                            "uninitialized while it still has readers");
}

TEST(Resource, CorrectSections) {
  auto resource = Resource::allocate("test", /*isInitialized=*/true);
  auto use1 =
      resource->beginUse("read1", kReadingResourceUse, ResourceSection(5, 10));
  auto use2 =
      resource->beginUse("read2", kReadingResourceUse, ResourceSection(10, 15));
  auto use3 =
      resource->beginUse("read3", kReadingResourceUse, ResourceSection(10, 15));
  use2.reset();
  use1.reset();
  auto use4 =
      resource->beginUse("read1", kReadingResourceUse, ResourceSection(0, 10));
  use3.reset();
  use4.reset();
  auto use5 =
      resource->beginUse("read1", kReadingResourceUse, ResourceSection(0, 15));
}

TEST(Resource, Resource_Overlapping_ExpectDeath) {
  auto resource = Resource::allocate("test", /*isInitialized=*/true);
  auto use1 =
      resource->beginUse("read1", kReadingResourceUse, ResourceSection(5, 10));
  auto use2 =
      resource->beginUse("read2", kReadingResourceUse, ResourceSection(10, 15));
  auto use3 =
      resource->beginUse("read3", kReadingResourceUse, ResourceSection(10, 15));
  EXPECT_DEATH_IF_SUPPORTED(
      resource->beginUse("read4", kReadingResourceUse, ResourceSection(4, 6)),
      "requested section for reading overlaps with existing section \\[5, "
      "10\\) "
      "for reading with active uses \\{'read1'\\}");
}

TEST(Resource, Resource_SectionReadWriteRace_ExpectDeath) {
  auto resource = Resource::allocate("test", /*isInitialized=*/true);
  auto use1 =
      resource->beginUse("read", kReadingResourceUse, ResourceSection(5, 10));
  EXPECT_DEATH_IF_SUPPORTED(
      resource->beginUse("write", kWritingResourceUse, ResourceSection(0, 20)),
      "requested section for writing overlaps with existing section "
      "\\[5, 10\\) for reading with active uses \\{'read'\\}");
}

} // namespace
