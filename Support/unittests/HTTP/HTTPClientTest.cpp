//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HTTP/HTTPClient.h"
#include "Support/FileSystemExtras.h"
#include "llvm/Support/Threading.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>

using namespace M;

namespace {

constexpr StringLiteral testData =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit";

} // namespace

static std::string pathToURL(const std::filesystem::path &path) {
  std::error_code ec;
  std::string pathStr = std::filesystem::absolute(path, ec).string();
  EXPECT_FALSE(ec);
  // Replace all backslashes with forward slashes for URI compatibility
  std::replace(pathStr.begin(), pathStr.end(), '\\', '/');

  return "file:///" + pathStr;
}

TempFile createTempFile() {
  std::error_code ec;
  std::filesystem::path tempDir = std::filesystem::temp_directory_path(ec);
  EXPECT_FALSE(ec);
  auto temp =
      TempFile::create((tempDir / "modular_payload_%%%%%%.txt").string());

  EXPECT_FALSE(temp.isError());
  // Open the file in output mode. If the file doesn't exist, it will be created
  std::ofstream outfile(temp->getPath());

  EXPECT_EQ(outfile.is_open(), true);
  outfile << testData.data();
  outfile.close(); // Remember to close the file when done
  return temp.takeValue();
}

// HACK: We can only initialize HTTPContext once per process but multiple test
// will need HTTPContext.
static HTTPContextRef GetHTTPContextRef() {
  static HTTPContextRef ref;
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() { ref = HTTPContext::init(); });
  return ref.copy();
}

TEST(ModularToolTest, fetchURL) {
  TempFile temp = createTempFile();
  std::error_code ec;
  auto filePath = std::filesystem::absolute(temp.getPath(), ec);
  EXPECT_FALSE(ec);
  auto urlPath = pathToURL(filePath);
  HTTPClient client(GetHTTPContextRef());
  const HTTPRequest request = {urlPath};
  std::string ostring;
  llvm::raw_string_ostream stream(ostring);
  auto result = client.executeRequest(request, stream);
  EXPECT_EQ(testData, ostring);
  EXPECT_EQ(result.isError(), false);
  EXPECT_EQ(std::filesystem::remove(filePath, ec), true);
  EXPECT_FALSE(ec);
}
