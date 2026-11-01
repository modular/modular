//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Target/lib/Support/StandardStatus.h"
#include "Target/lib/Support/Utils.h"

#include "HAL/M_driver_common.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstring>
#include <gtest/gtest.h>
#include <string>

using namespace M::Target;

//===----------------------------------------------------------------------===//
// copyStatusMessage(char*, int64_t, const char*)
//===----------------------------------------------------------------------===//

TEST(SupportTest, CopyStatusMessageConstCharEmpty) {
  char buf[16];
  copyStatusMessage(buf, sizeof(buf), "");
  EXPECT_STREQ(buf, "");
}

TEST(SupportTest, CopyStatusMessageConstCharShort) {
  char buf[16];
  copyStatusMessage(buf, sizeof(buf), "hello");
  EXPECT_STREQ(buf, "hello");
}

TEST(SupportTest, CopyStatusMessageConstCharExactFit) {
  const char *src = "12345678";
  char buf[9];
  copyStatusMessage(buf, 9, src);
  EXPECT_STREQ(buf, "12345678");
  EXPECT_EQ(buf[8], '\0');
}

TEST(SupportTest, CopyStatusMessageConstCharTruncated) {
  const char *src = "hello world";
  char buf[6];
  copyStatusMessage(buf, sizeof(buf), src);
  EXPECT_STREQ(buf, "hello");
  EXPECT_EQ(buf[5], '\0');
}

TEST(SupportTest, CopyStatusMessageConstCharSingleByteBuffer) {
  char buf[1];
  copyStatusMessage(buf, 1, "xyz");
  EXPECT_EQ(buf[0], '\0');
}

//===----------------------------------------------------------------------===//
// copyStatusMessage(char*, int64_t, llvm::Twine const&)
//===----------------------------------------------------------------------===//

TEST(SupportTest, CopyStatusMessageTwineFromLiteral) {
  char buf[32];
  copyStatusMessage(buf, sizeof(buf), llvm::Twine("twine literal"));
  EXPECT_STREQ(buf, "twine literal");
}

TEST(SupportTest, CopyStatusMessageTwineConcatenation) {
  char buf[32];
  copyStatusMessage(buf, sizeof(buf), llvm::Twine("foo") + "bar");
  EXPECT_STREQ(buf, "foobar");
}

TEST(SupportTest, CopyStatusMessageTwineTruncated) {
  char buf[5];
  copyStatusMessage(buf, sizeof(buf), llvm::Twine("truncated"));
  EXPECT_STREQ(buf, "trun");
  EXPECT_EQ(buf[4], '\0');
}

//===----------------------------------------------------------------------===//
// copyStatusMessage template (Stringish: std::string, llvm::StringRef)
//===----------------------------------------------------------------------===//

TEST(SupportTest, CopyStatusMessageStdStringShort) {
  char buf[32];
  std::string src("std::string");
  copyStatusMessage(buf, sizeof(buf), src);
  EXPECT_STREQ(buf, "std::string");
}

TEST(SupportTest, CopyStatusMessageStdStringLong) {
  std::string src(100, 'x');
  char buf[11];
  copyStatusMessage(buf, sizeof(buf), src);
  EXPECT_EQ(std::strlen(buf), 10u);
  EXPECT_EQ(buf[10], '\0');
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(buf[i], 'x');
}

TEST(SupportTest, CopyStatusMessageStringRef) {
  char buf[32];
  llvm::StringRef ref("stringref");
  copyStatusMessage(buf, sizeof(buf), ref);
  EXPECT_STREQ(buf, "stringref");
}

//===----------------------------------------------------------------------===//
// Standard status messages
//===----------------------------------------------------------------------===//

TEST(SupportTest, StandardStatusMessagesKnownStrings) {
  EXPECT_STREQ(standardStatusMessages[0], "Success");
  EXPECT_STREQ(standardStatusMessages[1], "Incompatible version");
  EXPECT_STREQ(standardStatusMessages[2], "Unknown error");
  EXPECT_STREQ(standardStatusMessages[3], "Argument out of range");
  EXPECT_STREQ(standardStatusMessages[4], "Unknown property name");
  EXPECT_STREQ(standardStatusMessages[5], "No devices");
  EXPECT_STREQ(standardStatusMessages[6], "Bad driver");
  EXPECT_STREQ(standardStatusMessages[7], "Bad device");
  EXPECT_STREQ(standardStatusMessages[8], "Bad context");
  EXPECT_STREQ(standardStatusMessages[9], "Bad memory");
  EXPECT_STREQ(standardStatusMessages[10], "Bad queue");
  EXPECT_STREQ(standardStatusMessages[11], "Bad event");
  EXPECT_STREQ(standardStatusMessages[12], "Bad function");
  EXPECT_STREQ(standardStatusMessages[13], "Out of memory");
  EXPECT_STREQ(standardStatusMessages[14], "Invalid argument");
  EXPECT_STREQ(standardStatusMessages[15], "Incompatible file type");
  EXPECT_STREQ(standardStatusMessages[16], "Bad bundle");
  EXPECT_STREQ(standardStatusMessages[17], "Execute mode not supported");
  EXPECT_STREQ(standardStatusMessages[18], "Unknown symbol");
  EXPECT_STREQ(standardStatusMessages[19], "Unsupported");
  EXPECT_STREQ(standardStatusMessages[20], "Unimplemented");
}

TEST(SupportTest, StandardStatusMessagesNonEmpty) {
  for (int64_t i = 0; i < M_DRIVER_NUM_STANDARD_STATUS_CODES; ++i) {
    ASSERT_NE(standardStatusMessages[i], nullptr)
        << "index " << i << " is nullptr";
    EXPECT_GT(std::strlen(standardStatusMessages[i]), 0u)
        << "index " << i << " is empty";
  }
}
